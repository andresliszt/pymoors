use proc_macro::TokenStream;
use quote::quote;
use quote::ToTokens;
use syn::parse::{Parse, ParseStream, Result};
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Fields, LitStr};

/// Structure for parsing the attribute arguments.
/// Now it expects only one string literal:
/// 1. The operator type ("mutation", "crossover", or "sampling")
struct PyOperatorArgs {
    operator_type: LitStr,
}

impl Parse for PyOperatorArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let operator_type: LitStr = input.parse()?;
        Ok(PyOperatorArgs { operator_type })
    }
}

/// Extracts the first non-empty doc string from the given attributes.
///
/// It heuristically finds the first and last double quotes in the token stream
/// (which should enclose the doc text) and returns the trimmed content.
/// (This approach is a bit hacky and may be improved in the future.)
fn extract_doc(attrs: &[Attribute]) -> String {
    for attr in attrs {
        if attr.path().is_ident("doc") {
            let tokens_str = attr.to_token_stream().to_string();
            if let (Some(start), Some(end)) = (tokens_str.find('\"'), tokens_str.rfind('\"')) {
                if end > start {
                    let doc = &tokens_str[start + 1..end];
                    if !doc.trim().is_empty() {
                        return doc.trim().to_string();
                    }
                }
            }
        }
    }
    String::new()
}

/// This attribute macro generates a Python binding wrapper for an operator
/// (mutation, crossover, or sampling). It creates a Python class with:
/// - A constructor (supporting unit structs or structs with named fields)
/// - Getters for each field (if any)
/// - An operator-specific method based on the provided operator type
///
/// Itakes only one argument (the operator type) and automatically
/// extracts the Rust doc comments from the original struct to use as the
/// Python class docstring.
///
/// # Usage
///
/// ```rust
/// #[py_operator("mutation")]
/// /// Mutation operator that flips bits in a binary individual with a specified mutation rate.
/// #[derive(Clone, Debug)]
/// pub struct BitFlipMutation {
///     pub gene_mutation_rate: f64,
/// }
/// ```
#[proc_macro_attribute]
pub fn py_operator(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the attribute arguments: only the operator type.
    let args = parse_macro_input!(attr as PyOperatorArgs);
    let operator_type = args.operator_type.value();

    // Parse the input as a DeriveInput.
    let input = parse_macro_input!(item as DeriveInput);
    // Extract the doc comments from the original struct.
    let combined_doc = extract_doc(&input.attrs);
    let original_ident = input.ident.clone();
    let py_class_name = original_ident.to_string();
    // Always generate the wrapper type as "Py" + original name.
    let py_wrapper_ident = syn::Ident::new(&format!("Py{}", original_ident), original_ident.span());

    // Extract fields: support named fields and unit structs.
    let fields_tokens: Vec<_> = if let Data::Struct(ref data) = input.data {
        match &data.fields {
            Fields::Named(ref fields_named) => fields_named.named.iter().collect(),
            Fields::Unit => Vec::new(),
            Fields::Unnamed(_) => {
                return syn::Error::new_spanned(
                    &original_ident,
                    "Only structs with named fields or unit structs are supported",
                )
                .to_compile_error()
                .into();
            }
        }
    } else {
        return syn::Error::new_spanned(
            &original_ident,
            "The py_operator attribute can only be applied to structs",
        )
        .to_compile_error()
        .into();
    };

    // Generate constructor arguments and field names.
    let constructor_args = fields_tokens.iter().map(|f| {
        let ident = f.ident.as_ref().unwrap();
        let ty = &f.ty;
        quote! { #ident: #ty }
    });
    let field_names = fields_tokens.iter().map(|f| {
        let ident = f.ident.as_ref().unwrap();
        quote! { #ident }
    });
    // Generate getters for each field.
    let getters = fields_tokens.iter().map(|f| {
        let ident = f.ident.as_ref().unwrap();
        let ty = &f.ty;
        quote! {
            #[getter]
            pub fn #ident(&self) -> #ty {
                self.inner.#ident.clone()
            }
        }
    });

    // Generate the operator-specific method.
    let operator_method = if operator_type == "mutation" {
        quote! {
            #[pyo3(signature = (population, seed=None))]
            pub fn mutate<'py>(
                &self,
                py: pyo3::prelude::Python<'py>,
                population: numpy::PyReadonlyArrayDyn<'py, f64>,
                seed: Option<u64>,
            ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, numpy::PyArray2<f64>>> {
                let owned_population = population.to_owned_array();
                let mut owned_population = owned_population.into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err("Population numpy array must be 2D to use mutate."))?;
                let mut rng = crate::random::MOORandomGenerator::new_from_seed(seed);
                self.inner.operate(&mut owned_population, 1.0, &mut rng);
                Ok(numpy::ToPyArray::to_pyarray(&owned_population, py))
            }
        }
    } else if operator_type == "crossover" {
        quote! {
            #[pyo3(signature = (parents_a, parents_b, seed=None))]
            pub fn crossover<'py>(
                &self,
                py: pyo3::prelude::Python<'py>,
                parents_a: numpy::PyReadonlyArrayDyn<'py, f64>,
                parents_b: numpy::PyReadonlyArrayDyn<'py, f64>,
                seed: Option<u64>,
            ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, numpy::PyArray2<f64>>> {
                let owned_parents_a = parents_a.to_owned_array();
                let owned_parents_b = parents_b.to_owned_array();
                let owned_parents_a = owned_parents_a.into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err("parent_a numpy array must be 2D to use crossover."))?;
                let owned_parents_b = owned_parents_b.into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err("parent_b numpy array must be 2D to use crossover."))?;
                let mut rng = crate::random::MOORandomGenerator::new_from_seed(seed);
                let offspring = self.inner.operate(&owned_parents_a, &owned_parents_b, 1.0, &mut rng);
                Ok(numpy::ToPyArray::to_pyarray(&offspring, py))
            }
        }
    } else if operator_type == "sampling" {
        quote! {
            #[pyo3(signature = (pop_size, n_vars, seed=None))]
            pub fn sample<'py>(
                &self,
                py: pyo3::prelude::Python<'py>,
                pop_size: usize,
                n_vars: usize,
                seed: Option<u64>,
            ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, numpy::PyArray2<f64>>> {
                let mut rng = crate::random::MOORandomGenerator::new_from_seed(seed);
                let sampled_population = self.inner.operate(pop_size, n_vars, &mut rng);
                Ok(numpy::ToPyArray::to_pyarray(&sampled_population, py))
            }
        }
    } else if operator_type == "duplicates" {
        quote! {
        #[pyo3(signature = (population, reference=None))]
        pub fn remove_duplicates<'py>(
                &self,
                py: pyo3::prelude::Python<'py>,
                population: numpy::PyReadonlyArrayDyn<'py, f64>,
                reference: Option<numpy::PyReadonlyArrayDyn<'py, f64>>,
        ) -> pyo3::prelude::PyResult<pyo3::prelude::Bound<'py, numpy::PyArray2<f64>>> {
                let population = population.to_owned_array();
                let population = population.into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("population numpy array must be 2D to use crossover."))?;
                let reference = reference
                .map(|ref_arr| {
                    ref_arr.to_owned_array().into_dimensionality::<ndarray::Ix2>()
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Reference numpy array must be 2D."))
                })
                .transpose()?;
                let clean_population = self.inner.remove(&population, reference.as_ref());
                Ok(numpy::ToPyArray::to_pyarray(&clean_population, py))
        }
        }
    } else {
        return syn::Error::new_spanned(
            &original_ident,
            "Unsupported operator type. Use \"mutation\", \"crossover\", or \"sampling\".",
        )
        .to_compile_error()
        .into();
    };

    let expanded = quote! {
        // Is there a way to avoid this import here?. Importing doesn't let us to use the macro twice in a module
        use numpy::PyArrayMethods;

        #input

        #[doc = #combined_doc]
        #[pyo3::prelude::pyclass(name = #py_class_name)]
        #[derive(Clone)]
        pub struct #py_wrapper_ident {
            pub inner: #original_ident,
        }

        #[pyo3::prelude::pymethods]
        impl #py_wrapper_ident {
            #[new]
            pub fn new(#(#constructor_args),*) -> Self {
                Self {
                    inner: #original_ident::new(#(#field_names),*),
                }
            }

            #(#getters)*

            #operator_method
        }
    };

    TokenStream::from(expanded)
}
