#[macro_export]
macro_rules! define_crossover_pyclass {
    // We expect something like:
    // define_crossover_pyclass!(ExponentialCrossover, PyExponentialCrossover, [cr: f64]);
    (
        $Name:ident,
        $PyName:ident,
        [ $( $arg:ident : $argty:ty ),* $(,)? ]
    ) => {

        #[pyclass(name = $Name)]
        #[derive(Clone)]
        pub struct $PyName {
            pub inner: $RustName,
        }

        #[pymethods]
        impl $PyName {
            #[new]
            fn new($( $arg : $argty ),*) -> Self {
                Self {
                    inner: $RustName::new($( $arg ),*),
                }
            }

            #[getter]
            fn name(&self) -> String {
                self.inner.name()
            }

            $(
                #[getter($arg)]
                fn $arg(&self) -> $argty {
                    let val = self.inner.$arg;
                    val
                }
            )*
        }
    }
}
