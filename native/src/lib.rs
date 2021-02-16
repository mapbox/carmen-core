use neon::prelude::*;
use rayon;

mod gridstore;
use gridstore::*;

mod fuzzy_phrase;
use crate::fuzzy_phrase::*;

register_module!(mut m, {
    // set thread count to 16 regardless of number of cores
    rayon::ThreadPoolBuilder::new().num_threads(16).build_global().unwrap();
    m.export_class::<JsGridStoreBuilder>("GridStoreBuilder")?;
    m.export_class::<JsGridStore>("GridStore")?;
    m.export_class::<JsGridKeyStoreKeyIterator>("GridStoreKeyIterator")?;
    m.export_function("coalesce", js_coalesce)?;
    m.export_function("stackable", js_stackable)?;
    m.export_function("stackAndCoalesce", js_stack_and_coalesce)?;

    m.export_class::<JsFuzzyPhraseSetBuilder>("FuzzyPhraseSetBuilder")?;
    m.export_class::<JsFuzzyPhraseSet>("FuzzyPhraseSet")?;

    Ok(())
});
