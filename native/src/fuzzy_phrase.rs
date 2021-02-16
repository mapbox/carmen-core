use neon::declare_types;
use neon::prelude::*;

use fuzzy_phrase::glue::{EndingType, FuzzyPhraseSet, FuzzyPhraseSetBuilder, WordReplacement};

declare_types! {
    pub class JsFuzzyPhraseSetBuilder as JsFuzzyPhraseSetBuilder for Option<FuzzyPhraseSetBuilder> {
        init(mut cx) {
            let filename = cx.argument::<JsString>(0)?.value();
            match FuzzyPhraseSetBuilder::new(filename) {
                Ok(s) => Ok(Some(s)),
                Err(e) => cx.throw_type_error(e.to_string())
            }
        }

        method insert(mut cx) {
            let js_phrase = cx.argument::<JsValue>(0)?;
            let phrase: Vec<String> = match neon_serde::from_value(&mut cx, js_phrase) {
                Ok(v) => v,
                Err(e) => return cx.throw_type_error(e.to_string())
            };

            let mut this = cx.this();

            let insert: Result<u32, String> = {
                let lock = cx.lock();
                let mut fp_builder = this.borrow_mut(&lock);
                match fp_builder.as_mut() {
                    Some(builder) => {
                        builder.insert(phrase.as_slice()).map_err(|e| e.to_string())
                    }
                    None => {
                        Err("unable to insert()".to_string())
                    }
                }
            };

            match insert {
                Ok(id) => Ok(JsNumber::new(&mut cx, id as f64).upcast()),
                Err(e) => cx.throw_type_error(e)
            }
        }

        method loadWordReplacements(mut cx) {
            let js_word_replacements = cx.argument::<JsValue>(0)?;
            let word_replacements: Vec<WordReplacement> = match neon_serde::from_value(&mut cx, js_word_replacements) {
                Ok(v) => v,
                Err(e) => return cx.throw_type_error(e.to_string())
            };

            let mut this = cx.this();

            let load: Result<(), String> = {
                let lock = cx.lock();
                let mut fp_builder = this.borrow_mut(&lock);
                match fp_builder.as_mut() {
                    Some(builder) => {
                        builder.load_word_replacements(word_replacements).map_err(|e| e.to_string())
                    }
                    None => {
                        Err("unable to load_word_replacements()".to_string())
                    }
                }
            };

            match load {
                Ok(()) => Ok(JsUndefined::new().upcast()),
                Err(e) => cx.throw_type_error(e)
            }
        }

        method finish(mut cx) {
            let mut this = cx.this();

            let finish = {
                let lock = cx.lock();
                let mut fp_builder = this.borrow_mut(&lock);
                match fp_builder.take() {
                    Some(builder) => builder.finish().map_err(|e| e.to_string()),
                    None => Err("unable to finish()".to_string())
                }
            };

            let convert = match finish {
                Ok(id_map) => {
                    let mut buffer = JsArrayBuffer::new(&mut cx, (id_map.len() * std::mem::size_of::<u32>()) as u32)?;

                    let lock = cx.lock();
                    let result = match buffer.try_borrow_mut(&lock) {
                        Ok(data) => {
                            let slice = data.as_mut_slice::<u32>();
                            slice.copy_from_slice(id_map.as_slice());

                            Ok(())
                        },
                        Err(e) => Err(e.to_string())
                    };
                    result.map(|_| buffer)
                },
                Err(e) => Err(e)
            };

            match convert {
                Ok(buffer) => Ok(buffer.upcast()),
                Err(e) => cx.throw_type_error(e)
            }
        }
    }

    pub class JsFuzzyPhraseSet as JsFuzzyPhraseSet for FuzzyPhraseSet {
        init(mut cx) {
            let filename = cx.argument::<JsString>(0)?.value();
            match FuzzyPhraseSet::from_path(filename) {
                Ok(s) => Ok(s),
                Err(e) => cx.throw_type_error(e.to_string())
            }
        }

        method contains(mut cx) {
            let js_phrase = cx.argument::<JsValue>(0)?;
            let phrase: Vec<String> = match neon_serde::from_value(&mut cx, js_phrase) {
                Ok(v) => v,
                Err(e) => return cx.throw_type_error(e.to_string())
            };

            let js_ending_type = cx.argument::<JsValue>(1)?;
            let ending_type: EndingType = match neon_serde::from_value(&mut cx, js_ending_type) {
                Ok(v) => v,
                Err(e) => return cx.throw_type_error(e.to_string())
            };

            let mut this = cx.this();

            let result = {
                let lock = cx.lock();
                let set = this.borrow_mut(&lock);

                set.contains(phrase.as_slice(), ending_type).map_err(|e| e.to_string())
            };

            match result {
                Ok(found) => Ok(JsBoolean::new(&mut cx, found).upcast()),
                Err(e) => cx.throw_type_error(e)
            }
        }

        method fuzzyMatch(mut cx) {
            let js_phrase = cx.argument::<JsValue>(0)?;
            let phrase: Vec<String> = match neon_serde::from_value(&mut cx, js_phrase) {
                Ok(v) => v,
                Err(e) => return cx.throw_type_error(e.to_string())
            };

            let max_word_dist = cx.argument::<JsNumber>(1)?.value() as u8;
            let max_phrase_dist = cx.argument::<JsNumber>(2)?.value() as u8;

            let js_ending_type = cx.argument::<JsValue>(3)?;
            let ending_type: EndingType = match neon_serde::from_value(&mut cx, js_ending_type) {
                Ok(v) => v,
                Err(e) => return cx.throw_type_error(e.to_string())
            };

            let mut this = cx.this();

            let result = {
                let lock = cx.lock();
                let set = this.borrow_mut(&lock);

                set.fuzzy_match(phrase.as_slice(), max_word_dist, max_phrase_dist, ending_type)
            };

            match result {
                Ok(matches) => {
                    match neon_serde::to_value(&mut cx, &matches) {
                        Ok(serialized) => Ok(serialized.upcast()),
                        Err(e) => cx.throw_type_error(e.to_string())
                    }
                },
                Err(e) => cx.throw_type_error(e.to_string())
            }
        }

        method fuzzyMatchWindows(mut cx) {
            let js_phrase = cx.argument::<JsValue>(0)?;
            let phrase: Vec<String> = match neon_serde::from_value(&mut cx, js_phrase) {
                Ok(v) => v,
                Err(e) => return cx.throw_type_error(e.to_string())
            };

            let max_word_dist = cx.argument::<JsNumber>(1)?.value() as u8;
            let max_phrase_dist = cx.argument::<JsNumber>(2)?.value() as u8;

            let js_ending_type = cx.argument::<JsValue>(3)?;
            let ending_type: EndingType = match neon_serde::from_value(&mut cx, js_ending_type) {
                Ok(v) => v,
                Err(e) => return cx.throw_type_error(e.to_string())
            };

            let mut this = cx.this();

            let result = {
                let lock = cx.lock();
                let set = this.borrow_mut(&lock);

                set.fuzzy_match_windows(phrase.as_slice(), max_word_dist, max_phrase_dist, ending_type)
            };

            match result {
                Ok(matches) => {
                    match neon_serde::to_value(&mut cx, &matches) {
                        Ok(serialized) => Ok(serialized.upcast()),
                        Err(e) => cx.throw_type_error(e.to_string())
                    }
                },
                Err(e) => cx.throw_type_error(e.to_string())
            }
        }

        method fuzzyMatchMulti(mut cx) {
            let js_phrases = cx.argument::<JsValue>(0)?;
            let phrases: Vec<(Vec<String>, EndingType)> = match neon_serde::from_value(&mut cx, js_phrases) {
                Ok(v) => v,
                Err(e) => return cx.throw_type_error(e.to_string())
            };

            let max_word_dist = cx.argument::<JsNumber>(1)?.value() as u8;
            let max_phrase_dist = cx.argument::<JsNumber>(2)?.value() as u8;

            let mut this = cx.this();

            let result = {
                let lock = cx.lock();
                let set = this.borrow_mut(&lock);

                set.fuzzy_match_multi(phrases.as_slice(), max_word_dist, max_phrase_dist)
            };

            match result {
                Ok(matches) => {
                    match neon_serde::to_value(&mut cx, &matches) {
                        Ok(serialized) => Ok(serialized.upcast()),
                        Err(e) => cx.throw_type_error(e.to_string())
                    }
                },
                Err(e) => cx.throw_type_error(e.to_string())
            }
        }

        method getByPhraseId(mut cx) {
            let phrase_id: u32 = cx.argument::<JsNumber>(0)?.value() as u32;

            let mut this = cx.this();

            let result = {
                let lock = cx.lock();
                let set = this.borrow_mut(&lock);

                set.get_by_phrase_id(phrase_id)
            };

            match result {
                Ok(Some(v)) => neon_serde::to_value(&mut cx, &v).or_else(|e| cx.throw_type_error(e.to_string())),
                Ok(None) => Ok(JsUndefined::new().upcast()),
                Err(e) => cx.throw_type_error(e.to_string())
            }
        }

        method getPrefixBins(mut cx) {
            let max_bin_size: usize = cx.argument::<JsNumber>(0)?.value() as usize;

            let mut this = cx.this();

            let result = {
                let lock = cx.lock();
                let set = this.borrow_mut(&lock);

                set.get_prefix_bins(max_bin_size)
            };

            match result {
                Ok(bins) => {
                    let mut bare_ids: Vec<u32> = bins.iter().map(|bin| bin.first.value() as u32).collect();
                    if let Some(bin) = bins.last() {
                        bare_ids.push(bin.last.value() as u32 + 1);
                    }

                    let mut buffer = JsArrayBuffer::new(&mut cx, (bare_ids.len() * std::mem::size_of::<u32>()) as u32)?;
                    let lock = cx.lock();
                    let result = match buffer.try_borrow_mut(&lock) {
                        Ok(data) => {
                            let slice = data.as_mut_slice::<u32>();
                            slice.copy_from_slice(bare_ids.as_slice());

                            Ok(())
                        },
                        Err(e) => Err(e.to_string())
                    };

                    match result {
                        Ok(()) => Ok(buffer.upcast()),
                        Err(e) => cx.throw_type_error(e.to_string())
                    }
                },
                Err(e) => cx.throw_type_error(e.to_string())
            }
        }
    }
}
