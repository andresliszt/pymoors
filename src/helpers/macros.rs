#[macro_export]
macro_rules! unwrap_operator {
    (
        $fn_name:ident,
        $trait_type:path,
        $error_str:expr,
        [ $( $py_type:path ),+ $(,)? ]
    ) => {
        pub fn $fn_name(py_obj: PyObject) -> PyResult<Box<dyn $trait_type>> {
            Python::with_gil(|py| {
                $(
                    if let Ok(extracted) = py_obj.extract::<$py_type>(py) {
                        return Ok(Box::new(extracted.inner) as Box<dyn $trait_type>);
                    }
                )+
                Err(PyValueError::new_err($error_str))
            })
        }
    };
}
