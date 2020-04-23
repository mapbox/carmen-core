use carmen_core::gridstore::{coalesce, stackable, stack_and_coalesce};
use carmen_core::gridstore::{
    CoalesceContext, GridEntry, GridKey, GridStore, GridStoreBuilder, MatchOpts, MatchKey, MatchKeyWithId, PhrasematchSubquery
};

use neon::prelude::*;
use neon::{class_definition, declare_types, impl_managed, register_module};
use neon_serde::errors::Result as LibResult;
use serde::Deserialize;
use owning_ref::OwningHandle;
use failure::Error;
use fixedbitset::FixedBitSet;

use std::sync::Arc;

type ArcGridStore = Arc<GridStore>;

struct CoalesceTask {
    argument: (Vec<PhrasematchSubquery<ArcGridStore>>, MatchOpts),
}

impl Task for CoalesceTask {
    type Output = Vec<CoalesceContext>;
    type Error = String;
    type JsEvent = JsArray;

    fn perform(&self) -> Result<Vec<CoalesceContext>, String> {
        coalesce(self.argument.0.clone(), &self.argument.1).map_err(|err| err.to_string())
    }

    fn complete<'a>(
        self,
        mut cx: TaskContext<'a>,
        result: Result<Vec<CoalesceContext>, String>,
    ) -> JsResult<JsArray> {
        let converted_result = {
            match &result {
                Ok(r) => r,
                Err(s) => return cx.throw_error(s),
            }
        };
        Ok(neon_serde::to_value(&mut cx, converted_result)?
            .downcast::<JsArray>()
            .or_throw(&mut cx)?)
    }
}

struct StackAndCoalesceTask {
    argument: (Vec<PhrasematchSubquery<ArcGridStore>>, MatchOpts),
}

impl Task for StackAndCoalesceTask {
    type Output = Vec<CoalesceContext>;
    type Error = String;
    type JsEvent = JsArray;

    fn perform(&self) -> Result<Vec<CoalesceContext>, String> {
        stack_and_coalesce(&self.argument.0, &self.argument.1).map_err(|err| err.to_string())
    }

    fn complete<'a>(
        self,
        mut cx: TaskContext<'a>,
        result: Result<Vec<CoalesceContext>, String>,
    ) -> JsResult<JsArray> {
        let converted_result = {
            match &result {
                Ok(r) => r,
                Err(s) => return cx.throw_error(s),
            }
        };
        Ok(neon_serde::to_value(&mut cx, converted_result)?
            .downcast::<JsArray>()
            .or_throw(&mut cx)?)
    }
}

type KeyIterator = OwningHandle<ArcGridStore, Box<dyn Iterator<Item=Result<GridKey, Error>>>>;

#[derive(Deserialize, Debug, PartialEq, Clone)]
struct GridStoreOpts {
    pub zoom: u16,
    pub type_id: u16,
    pub coalesce_radius: f64,
}

declare_types! {
    pub class JsGridStoreBuilder as JsGridStoreBuilder for Option<GridStoreBuilder> {
        init(mut cx) {
            let filename = cx.argument::<JsString>(0)?.value();
            match GridStoreBuilder::new(filename) {
                Ok(s) => Ok(Some(s)),
                Err(e) => cx.throw_type_error(e.to_string())
            }
        }

        method insert(mut cx) {
            let (key, values) = prep_for_insert(&mut cx)?;
            let mut this = cx.this();

            // lock falls out of scope at the end of this block
            // in order to be able to borrow `cx` for the error block we assign it to a variable

            let insert: Result<(), String> = {
                let lock = cx.lock();
                let mut gridstore = this.borrow_mut(&lock);
                match gridstore.as_mut() {
                    Some(builder) => {
                        builder.insert(&key, values).map_err(|e| e.to_string())
                    }
                    None => {
                        Err("unable to insert()".to_string())
                    }
                }
            };

            match insert {
                Ok(_) => Ok(JsUndefined::new().upcast()),
                Err(e) => cx.throw_type_error(e)
            }
        }

        method append(mut cx) {
            let (key, values) = prep_for_insert(&mut cx)?;
            let mut this = cx.this();

            // lock falls out of scope at the end of this block
            // in order to be able to borrow `cx` for the error block we assign it to a variable

            let insert: Result<(), String> = {
                let lock = cx.lock();
                let mut gridstore = this.borrow_mut(&lock);
                match gridstore.as_mut() {
                    Some(builder) => {
                        builder.append(&key, values).map_err(|e| e.to_string())
                    }
                    None => {
                        Err("unable to insert()".to_string())
                    }
                }
            };

            match insert {
                Ok(_) => Ok(JsUndefined::new().upcast()),
                Err(e) => cx.throw_type_error(e)
            }
        }

        method compactAppend(mut cx) {
            let grid_key = cx.argument::<JsObject>(0)?;
            let phrase_id: u32 = grid_key
                .get(&mut cx, "phrase_id")?
                .downcast::<JsNumber>()
                .or_throw(&mut cx)?
                .value() as u32;

            let js_lang_set = grid_key.get(&mut cx, "lang_set")?;
            let lang_set: u128 = langarray_to_langset(&mut cx, js_lang_set)?;

            let key = GridKey { phrase_id, lang_set };

            let relev = cx.argument::<JsNumber>(1)?.value() as f64;
            let score = cx.argument::<JsNumber>(2)?.value() as u8;
            let id = cx.argument::<JsNumber>(3)?.value() as u32;
            let source_phrase_hash = cx.argument::<JsNumber>(4)?.value() as u8;
            let js_coords = cx.argument::<JsValue>(5)?;
            let coords: Vec<(u16, u16)> = neon_serde::from_value(&mut cx, js_coords)?;

            let mut this = cx.this();

            let insert: Result<(), String> = {
                let lock = cx.lock();
                let mut gridstore = this.borrow_mut(&lock);
                match gridstore.as_mut() {
                    Some(builder) => {
                        Ok(builder.compact_append(&key, relev, score, id, source_phrase_hash, &coords))
                    }
                    None => {
                        Err("unable to insert()".to_string())
                    }
                }
            };

            match insert {
                Ok(_) => Ok(JsUndefined::new().upcast()),
                Err(e) => cx.throw_type_error(e)
            }
        }

        method renumber(mut cx) {
            let js_id_map = cx.argument::<JsArrayBuffer>(0)?;
            let mut this = cx.this();

            let result: Result<(), String> = {
                let lock = cx.lock();

                let borrow_result = match js_id_map.try_borrow(&lock) {
                    Ok(data) => {
                        let slice = data.as_slice::<u32>();

                        let mut gridstore = this.borrow_mut(&lock);
                        match gridstore.as_mut() {
                            Some(builder) => {
                                builder.renumber(slice).map_err(|e| e.to_string())
                            }
                            None => {
                                Err("can't call renumber after finish()".to_owned())
                            }
                        }
                    },
                    Err(e) => Err(e.to_string())
                };

                borrow_result
            };

            match result {
                Ok(_) => Ok(JsUndefined::new().upcast()),
                Err(e) => cx.throw_type_error(e)
            }
        }

        method loadBinBoundaries(mut cx) {
            let bin_boundaries = cx.argument::<JsArrayBuffer>(0)?;
            let mut this = cx.this();

            let result: Result<(), String> = {
                let lock = cx.lock();

                let borrow_result = match bin_boundaries.try_borrow(&lock) {
                    Ok(data) => {
                        let slice = data.as_slice::<u32>();

                        let mut gridstore = this.borrow_mut(&lock);
                        match gridstore.as_mut() {
                            Some(builder) => {
                                builder.load_bin_boundaries(slice.to_vec()).map_err(|e| e.to_string())
                            }
                            None => {
                                Err("can't call loadBinBoundaries after finish()".to_owned())
                            }
                        }
                    },
                    Err(e) => Err(e.to_string())
                };

                borrow_result
            };

            match result {
                Ok(_) => Ok(JsUndefined::new().upcast()),
                Err(e) => cx.throw_type_error(e)
            }
        }

        method finish(mut cx) {
            let mut this = cx.this();

            let finish: Result<(), String> = {
                let lock = cx.lock();
                let mut gridstore = this.borrow_mut(&lock);
                match gridstore.take() {
                    Some(builder) => {
                        builder.finish().map_err(|e| e.to_string())
                    }
                    None => {
                        Err("unable to finish()".to_string())
                    }
                }
            };

            match finish {
                Ok(_) => Ok(JsUndefined::new().upcast()),
                Err(e) => cx.throw_type_error(e)
            }
        }
    }

    pub class JsGridStore as JsGridStore for ArcGridStore {
        init(mut cx) {
            let filename = cx.argument::<JsString>(0)?.value();
            let store = match cx.argument_opt(1) {
                Some(arg) => {
                    let opts: GridStoreOpts = neon_serde::from_value(&mut cx, arg)?;

                    GridStore::new_with_options(
                        filename,
                        opts.zoom,
                        opts.type_id,
                        opts.coalesce_radius
                    )
                },
                None => GridStore::new(filename)
            };
            match store {
                Ok(s) => Ok(Arc::new(s)),
                Err(e) => cx.throw_type_error(e.to_string())
            }
        }

        method get(mut cx) {
            let grid_key = cx.argument::<JsObject>(0)?;

            let phrase_id: u32 = grid_key
                .get(&mut cx, "phrase_id")?
                .downcast::<JsNumber>()
                .or_throw(&mut cx)?
                .value() as u32;

            let js_lang_set = grid_key.get(&mut cx, "lang_set")?;
            let lang_set: u128 = langarray_to_langset(&mut cx, js_lang_set)?;

            let key = GridKey { phrase_id, lang_set };

            let mut this = cx.this();

            let result = {
                let lock = cx.lock();
                let grid_store = this.borrow_mut(&lock);

                grid_store.get(&key).map(|option| option.map(|iter| iter.collect::<Vec<_>>()))
            };

            match result {
                Ok(Some(v)) => Ok(neon_serde::to_value(&mut cx, &v)?),
                Ok(None) => Ok(JsUndefined::new().upcast()),
                Err(e) => cx.throw_type_error(e.to_string())
            }
        }
    }

    pub class JsGridKeyStoreKeyIterator as JsGridKeyStoreKeyIterator for KeyIterator {
        init(mut cx) {
            let js_gridstore = cx.argument::<JsGridStore>(0)?;
            let gridstore = {
                let guard = cx.lock();
                // shallow clone of the Arc
                let gridstore_clone = js_gridstore.borrow(&guard).clone();
                gridstore_clone
            };

            Ok(OwningHandle::new_with_fn(gridstore, |gs| {
                // this is per the OwningHandle docs -- the handle keeps both the arc and the
                // iterator, so the former is guaranteed to be around as long as the latter
                let gridstore = unsafe { &*gs };
                let iter: Box<dyn Iterator<Item=Result<GridKey, Error>>> = Box::new(gridstore.keys());
                iter
            }))
        }

        method next(mut cx) {
            let mut this = cx.this();

            let next_gk = {
                let lock = cx.lock();
                let mut iter = this.borrow_mut(&lock);

                iter.next()
            };

            match next_gk {
                Some(Ok(gk)) => {
                    let out = JsObject::new(&mut cx);

                    let done_label = JsString::new(&mut cx, "done");
                    let done_value = JsBoolean::new(&mut cx, false);
                    out.set(&mut cx, done_label, done_value)?;

                    let value_label = JsString::new(&mut cx, "value");
                    let js_gk = JsObject::new(&mut cx);
                    out.set(&mut cx, value_label, js_gk)?;

                    let phrase_id_label = JsString::new(&mut cx, "phrase_id");
                    let phrase_id_value = JsNumber::new(&mut cx, gk.phrase_id);
                    js_gk.set(&mut cx, phrase_id_label, phrase_id_value)?;

                    let lang_set_label = JsString::new(&mut cx, "lang_set");
                    let lang_set_value = langset_to_langarray(&mut cx, gk.lang_set);
                    js_gk.set(&mut cx, lang_set_label, lang_set_value)?;

                    Ok(out.upcast())
                }
                Some(Err(e)) => {
                    cx.throw_type_error(e.to_string())
                }
                None => {
                    let out = JsObject::new(&mut cx);
                    let done_label = JsString::new(&mut cx, "done");
                    let done_value = JsBoolean::new(&mut cx, true);
                    out.set(&mut cx, done_label, done_value)?;
                    Ok(out.upcast())
                }
            }
        }
    }
}

fn langarray_to_langset<'j, C>(cx: &mut C, maybe_lang_array: Handle<'j, JsValue>) -> Result<u128, neon_serde::errors::Error>
where
    C: Context<'j>,
{
    if let Ok(lang_array) = maybe_lang_array.downcast::<JsArray>() {
        let mut out = 0u128;
        for i in 0..lang_array.len() {
            let converted_lang_array = lang_array.get(cx, i)?.downcast::<JsNumber>().or_throw(cx)?.value() as usize;
            if  converted_lang_array >= 128 {
                continue;
            } else {
            out = out
                | (1 << converted_lang_array);
            }
        }
        Ok(out)
    } else if let Ok(_) = maybe_lang_array.downcast::<JsNull>() {
        Ok(std::u128::MAX)
    } else if let Ok(_) = maybe_lang_array.downcast::<JsUndefined>() {
        Ok(std::u128::MAX)
    } else {
        cx.throw_type_error("Expected array, undefined, or null for lang_set")?
    }
}

fn langset_to_langarray<'j, C: Context<'j>>(cx: &mut C, lang_set: u128) -> Handle<'j, JsArray> {
    let out = JsArray::new(cx, 0);
    let mut i = 0;
    for j in 0..128 {
        let bit = 1u128 << j;
        if lang_set & bit != 0 {
            let num = JsNumber::new(cx, j);
            out.set(cx, i, num).expect("failed to set array slot");
            i += 1;
        }
    }
    out
}

pub fn js_coalesce(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let js_phrase_subq = { cx.argument::<JsArray>(0)? };
    let js_match_ops = { cx.argument::<JsValue>(1)? };
    let phrase_subq: Vec<PhrasematchSubquery<ArcGridStore>> =
        deserialize_phrasesubq(&mut cx, js_phrase_subq)?;
    let match_opts: MatchOpts = neon_serde::from_value(&mut cx, js_match_ops)?;
    let cb = cx.argument::<JsFunction>(2)?;

    let task = CoalesceTask { argument: (phrase_subq, match_opts) };
    task.schedule(cb);

    Ok(cx.undefined())
}

pub fn js_stack_and_coalesce(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let js_phrase_subq = { cx.argument::<JsArray>(0)? };
    let js_match_ops = { cx.argument::<JsValue>(1)? };
    let phrase_subq: Vec<PhrasematchSubquery<ArcGridStore>> =
        deserialize_phrasesubq(&mut cx, js_phrase_subq)?;
    let match_opts: MatchOpts = neon_serde::from_value(&mut cx, js_match_ops)?;
    let cb = cx.argument::<JsFunction>(2)?;

    let task = StackAndCoalesceTask { argument: (phrase_subq, match_opts) };
    task.schedule(cb);

    Ok(cx.undefined())
}

fn deserialize_phrasesubq<'j, C>(
    cx: &mut C,
    js_phrase_subq_array: Handle<'j, JsArray>,
) -> LibResult<Vec<PhrasematchSubquery<ArcGridStore>>>
where
    C: Context<'j>,
{
    let mut phrasematches: Vec<PhrasematchSubquery<ArcGridStore>> =
        Vec::with_capacity(js_phrase_subq_array.len() as usize);
    for i in 0..js_phrase_subq_array.len() {
        let js_phrasematch =
            js_phrase_subq_array.get(cx, i)?.downcast::<JsObject>().or_throw(cx)?;
        let js_gridstore =
            js_phrasematch.get(cx, "store")?.downcast::<JsGridStore>().or_throw(cx)?;
        let gridstore = {
            let guard = cx.lock();
            // shallow clone of the Arc
            let gridstore_clone = js_gridstore.borrow(&guard).clone();
            gridstore_clone
        };
        let weight = js_phrasematch.get(cx, "weight")?;
        let mask = js_phrasematch.get(cx, "mask")?;

        let match_key = js_phrasematch.get(cx, "match_key")?.downcast::<JsObject>().or_throw(cx)?;
        let match_phrase = match_key.get(cx, "match_phrase")?;

        let js_lang_set = match_key.get(cx, "lang_set")?;
        let lang_set: u128 = langarray_to_langset(cx, js_lang_set)?;

        let id = js_phrasematch.get(cx, "id")?;

        let idx = js_phrasematch.get(cx, "idx")?;

        let js_non_overlapping_indexes = js_phrasematch.get(cx, "non_overlapping_indexes")?;
        let non_overlapping_indexes: Vec<u32> = neon_serde::from_value(cx, js_non_overlapping_indexes)?;

        let subq = PhrasematchSubquery {
            store: gridstore,
            weight: neon_serde::from_value(cx, weight)?,
            match_keys: vec![MatchKeyWithId {
                key: MatchKey { match_phrase: neon_serde::from_value(cx, match_phrase)?, lang_set },
                id: neon_serde::from_value(cx, id)?,
            }],
            mask: neon_serde::from_value(cx, mask)?,
            idx: neon_serde::from_value(cx, idx)?,
            non_overlapping_indexes: vector_to_fixedbitset(non_overlapping_indexes),
        };
        phrasematches.push(subq);
    }
    Ok(phrasematches)
}

pub fn vector_to_fixedbitset(bitset: Vec<u32>) -> FixedBitSet {
    let mut fbs = FixedBitSet::with_capacity(128);
    for bit in bitset {
        if bit == 1 {
            fbs.insert(1);
        } else {
            fbs.insert(0);
        }
    }
    return fbs;
}

pub fn js_stackable(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let js_phrasematch_result = { cx.argument::<JsArray>(0)? };
    let phrasematch_results: Vec<PhrasematchSubquery<ArcGridStore>> =
        deserialize_phrasesubq(&mut cx, js_phrasematch_result)?;
    stackable(&phrasematch_results);

    Ok(cx.undefined())
}

#[inline(always)]
fn prep_for_insert<'j, T: neon::object::This>(cx: &mut CallContext<'j, T>) -> Result<(GridKey, Vec<GridEntry>), neon_serde::errors::Error> {
    let grid_key = cx.argument::<JsObject>(0)?;
    let grid_entry = cx.argument::<JsValue>(1)?;
    let values: Vec<GridEntry> = neon_serde::from_value(cx, grid_entry)?;
    let phrase_id: u32 = grid_key
        .get(cx, "phrase_id")?
        .downcast::<JsNumber>()
        .or_throw(cx)?
        .value() as u32;

    let js_lang_set = grid_key.get(cx, "lang_set")?;
    let lang_set: u128 = langarray_to_langset(cx, js_lang_set)?;

    let key = GridKey { phrase_id, lang_set };

    Ok((key, values))
}

register_module!(mut m, {
    m.export_class::<JsGridStoreBuilder>("GridStoreBuilder")?;
    m.export_class::<JsGridStore>("GridStore")?;
    m.export_class::<JsGridKeyStoreKeyIterator>("GridStoreKeyIterator")?;
    m.export_function("coalesce", js_coalesce)?;
    m.export_function("stackable", js_stackable)?;
    m.export_function("stackAndCoalesce", js_stack_and_coalesce)?;
    Ok(())
});
