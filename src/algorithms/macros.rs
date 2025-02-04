#[macro_export]
macro_rules! define_multiobj_pyclass {
    (
        $RustStructName:ident,
        $PyStructName:ident,
        $PyClassName:literal
    ) => {
        use numpy::ToPyArray;
        use pyo3::types::PyDict;

        use crate::algorithms::MultiObjectiveAlgorithm;
        use crate::genetic::{PopulationConstraints, PopulationFitness, PopulationGenes};
        use crate::helpers::duplicates::PopulationCleaner;
        use crate::operators::{
            CrossoverOperator, MutationOperator, SamplingOperator, SelectionOperator,
            SurvivalOperator,
        };

        // The Rust struct that wraps MultiObjectiveAlgorithm
        pub struct $RustStructName {
            pub algorithm: MultiObjectiveAlgorithm,
        }

        impl $RustStructName {
            pub fn new(
                sampler: Box<dyn SamplingOperator>,
                crossover: Box<dyn CrossoverOperator>,
                mutation: Box<dyn MutationOperator>,
                selector: Box<dyn SelectionOperator>,
                survivor: Box<dyn SurvivalOperator>,
                duplicates_cleaner: Option<Box<dyn PopulationCleaner>>,
                fitness_fn: Box<dyn Fn(&PopulationGenes) -> PopulationFitness>,
                n_vars: usize,
                pop_size: usize,
                n_offsprings: usize,
                num_iterations: usize,
                mutation_rate: f64,
                crossover_rate: f64,
                keep_infeasible: bool,
                constraints_fn: Option<Box<dyn Fn(&PopulationGenes) -> PopulationConstraints>>,
                // Optional lower bound for each gene.
                lower_bound: Option<f64>,
                // Optional upper bound for each gene.
                upper_bound: Option<f64>,
            ) -> Self {
                // Build the shared MultiObjectiveAlgorithm
                let algorithm = MultiObjectiveAlgorithm::new(
                    sampler,
                    selector,
                    survivor,
                    crossover,
                    mutation,
                    duplicates_cleaner,
                    fitness_fn,
                    n_vars,
                    pop_size,
                    n_offsprings,
                    num_iterations,
                    mutation_rate,
                    crossover_rate,
                    keep_infeasible,
                    constraints_fn,
                    lower_bound,
                    upper_bound,
                );

                Self { algorithm }
            }

            pub fn run(&mut self) {
                self.algorithm.run();
            }
        }

        // The PyO3-exposed struct
        #[pyclass(name = $PyClassName, unsendable)]
        pub struct $PyStructName {
            pub inner: $RustStructName,
        }

        // Implement PyO3 methods
        #[pymethods]
        impl $PyStructName {
            pub fn run(&mut self) {
                self.inner.run();
            }

            // The population getter
            #[getter]
            pub fn population(&self, py: Python) -> PyResult<PyObject> {
                // Here we factor out the repeated code into a
                // function or do it inline. For brevity, inline here:
                let pydantic_module = py.import("pymoors.schemas")?;
                let population_class = pydantic_module.getattr("Population")?;

                let population = &self.inner.algorithm.population;
                let py_genes = population.genes.to_pyarray(py);
                let py_fitness = population.fitness.to_pyarray(py);
                let py_rank = population.rank.to_pyarray(py);
                let py_constraints = if let Some(ref c) = population.constraints {
                    c.to_pyarray(py).into_py(py)
                } else {
                    py.None().into_py(py)
                };

                let kwargs = PyDict::new(py);
                kwargs.set_item("genes", py_genes)?;
                kwargs.set_item("fitness", py_fitness)?;
                kwargs.set_item("rank", py_rank)?;
                kwargs.set_item("constraints", py_constraints)?;

                let pydantic_instance = population_class.call((), Some(&kwargs))?;
                Ok(pydantic_instance.into_py(py))
            }
        }
    };
}
