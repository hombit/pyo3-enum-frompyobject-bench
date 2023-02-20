use criterion::{black_box, criterion_group, criterion_main, Criterion};
use numpy::{Element, PyReadonlyArray1};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet};

use pyo3_enum_frompyobject_bench::{Collection, GenericFloatArray1};

fn extract_concrete_array<T: Element>(py: Python, array: PyObject) -> PyResult<()> {
    let array: PyReadonlyArray1<T> = array.extract(py)?;
    black_box(array);
    Ok(())
}

fn extract_generic_array(py: Python, array: PyObject) -> PyResult<()> {
    let array: GenericFloatArray1 = array.extract(py)?;
    black_box(array);
    Ok(())
}

fn extract_generic_array_ifelse(py: Python, array: PyObject) -> PyResult<()> {
    let array = if let Ok(array_f32) = array.extract(py) {
        GenericFloatArray1::Float32(array_f32)
    } else if let Ok(array_f64) = array.extract(py) {
        GenericFloatArray1::Float64(array_f64)
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Expected a numpy array with dtype float32 or float64",
        ));
    };
    black_box(array);
    Ok(())
}

fn benchmark_numpy(c: &mut Criterion) {
    println!("Running benchmark");
    Python::with_gil(|py| {
        let np = py.import("numpy").unwrap();

        let array_f32: PyObject = {
            let array: [f32; 0] = [];
            let py_any = np
                .getattr("array")
                .unwrap()
                .call((array, "float32"), None)
                .unwrap();
            py_any.into()
        };
        c.bench_function("extract concrete f32 array", |b| {
            b.iter(|| extract_concrete_array::<f32>(py, array_f32.clone()).unwrap())
        });
        c.bench_function("extract generic f32 array", |b| {
            b.iter(|| extract_generic_array(py, array_f32.clone()).unwrap())
        });
        c.bench_function("extract generic f32 array manually", |b| {
            b.iter(|| extract_generic_array_ifelse(py, array_f32.clone()).unwrap())
        });

        let array_f64: PyObject = {
            let array: [f64; 0] = [];
            let py_any = np
                .getattr("array")
                .unwrap()
                .call((array, "float64"), None)
                .unwrap();
            py_any.into()
        };
        c.bench_function("extract concrete f64 array", |b| {
            b.iter(|| extract_concrete_array::<f64>(py, array_f64.clone()).unwrap())
        });
        c.bench_function("extract generic f64 array", |b| {
            b.iter(|| extract_generic_array(py, array_f64.clone()).unwrap())
        });
        c.bench_function("extract generic f64 array manually", |b| {
            b.iter(|| extract_generic_array_ifelse(py, array_f64.clone()).unwrap())
        });
    })
}

fn extract_concrete_collection<'py, T>(py: Python<'py>, collection: &'py PyObject) -> PyResult<()>
where
    T: 'static + PyTryFrom<'py>,
{
    let collection: &T = collection.downcast(py)?;
    black_box(collection);
    Ok(())
}

fn extract_generic_collection(py: Python, collection: PyObject) -> PyResult<()> {
    let collection: Collection = collection.extract(py)?;
    black_box(collection);
    Ok(())
}

fn extract_generic_collection_ifelse(py: Python, collection: PyObject) -> PyResult<()> {
    let collection = if let Ok(collection_list) = collection.extract(py) {
        Collection::List(collection_list)
    } else if let Ok(collection_tuple) = collection.extract(py) {
        Collection::Tuple(collection_tuple)
    } else if let Ok(collection_set) = collection.extract(py) {
        Collection::Set(collection_set)
    } else {
        return Err(PyErr::new::<PyTypeError, _>(
            "Expected a list, tuple or set",
        ));
    };
    black_box(collection);
    Ok(())
}

fn benchmark_collection(c: &mut Criterion) {
    Python::with_gil(|py| {
        let list: PyObject = PyList::empty(py).into();
        c.bench_function("extract concrete list", |b| {
            b.iter(|| extract_concrete_collection::<PyList>(py, &list).unwrap())
        });
        c.bench_function("extract generic list", |b| {
            b.iter(|| extract_generic_collection(py, list.clone()).unwrap())
        });
        c.bench_function("extract generic list manually", |b| {
            b.iter(|| extract_generic_collection_ifelse(py, list.clone()).unwrap())
        });

        let set: PyObject = PySet::empty(py).unwrap().to_object(py);
        c.bench_function("extract concrete set", |b| {
            b.iter(|| extract_concrete_collection::<PySet>(py, &set).unwrap())
        });
        c.bench_function("extract generic set", |b| {
            b.iter(|| extract_generic_collection(py, set.clone()).unwrap())
        });
        c.bench_function("extract generic set manually", |b| {
            b.iter(|| extract_generic_collection_ifelse(py, set.clone()).unwrap())
        });
    })
}

criterion_group!(benches, benchmark_collection, benchmark_numpy);
criterion_main!(benches);
