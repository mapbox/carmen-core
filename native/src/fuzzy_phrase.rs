#[macro_use]
extern crate neon;
extern crate fuzzy_phrase;
extern crate neon_serde;

use neon::mem::Handle;
use neon::vm::{This, Lock, FunctionCall, JsResult};
use neon::js::{JsFunction, Object, JsString, Value, JsUndefined, JsArray, JsBoolean, JsNumber, JsInteger};
use neon::js::binary::JsArrayBuffer;
use neon::js::class::{JsClass, Class};
use neon::js::error::{Kind, JsError};

use fuzzy_phrase::glue::{FuzzyPhraseSetBuilder, FuzzyPhraseSet, WordReplacement, EndingType};

trait CheckArgument {
    fn check_argument<V: Value>(&mut self, i: i32) -> JsResult<V>;
}

impl<'a, T: This> CheckArgument for FunctionCall<'a, T> {
    fn check_argument<V: Value>(&mut self, i: i32) -> JsResult<V> {
        self.arguments.require(self.scope, i)?.check::<V>()
    }
}

declare_types! {
    pub class JsFuzzyPhraseSetBuilder as JsFuzzyPhraseSetBuilder for Option<FuzzyPhraseSetBuilder> {
        init(mut call) {
            let filename = call
                .check_argument::<JsString>(0)
                ?.value();
            match FuzzyPhraseSetBuilder::new(filename){
                Ok(response) => {
                    Ok(Some(response))
                },
                Err(e) => {
                    println!("{:?}", e);
                    JsError::throw(Kind::TypeError, e.description())
                }
            }
        }

        method insert(call) {
            let phrase_array = call.arguments.require(call.scope, 0)?.check::<JsArray>()?;

            let mut v: Vec<String> = Vec::new();

            for i in 0..phrase_array.len() {
                let string = phrase_array.get(call.scope, i)
                ?.check::<JsString>()
                ?.value();

                v.push(string);
            }

            let mut this: Handle<JsFuzzyPhraseSetBuilder> = call.arguments.this(call.scope);

            let result = this.grab(|fuzzyphrasesetbuilder| {
                match fuzzyphrasesetbuilder {
                    Some(builder) => {
                        match builder.insert(v.as_slice()) {
                            Ok(tmp_id) => Ok(tmp_id),
                            Err(e) => {
                                println!("{:?}", e);
                                JsError::throw(Kind::TypeError, e.description())
                            }
                        }
                    },
                    None => {
                        JsError::throw(Kind::TypeError, "unable to insert()")
                    }
                }
            });

            Ok(JsNumber::new(call.scope, result? as f64).upcast())
        }

        method loadWordReplacements(call) {
            let scope = call.scope;
            let mut this: Handle<JsFuzzyPhraseSetBuilder> = call.arguments.this(scope);
            let word_array = call.arguments.require(scope, 0)?;
            let word_replacements: Vec<WordReplacement> = neon_serde::from_value(scope, word_array)?;

            this.grab(|fuzzyphrasesetbuilder| {
                match fuzzyphrasesetbuilder {
                    Some(builder) => {
                        match builder.load_word_replacements(word_replacements) {
                            Ok(()) => Ok(JsUndefined::new().upcast()),
                            Err(e) => {
                                println!("{:?}", e);
                                JsError::throw(Kind::TypeError, e.description())
                            }
                        }
                    },
                    None => {
                        JsError::throw(Kind::TypeError, "unable to load_word_replacements()")
                    }
                }
            })
        }

        method finish(call) {
            let scope = call.scope;
            let mut this: Handle<JsFuzzyPhraseSetBuilder> = call.arguments.this(scope);

            let result = this.grab(|fuzzyphrasesetbuilder| {
                match fuzzyphrasesetbuilder.take() {
                    Some(builder) => {
                        match builder.finish() {
                            Ok(id_map) => Ok(id_map),
                            Err(e) => {
                                println!("{:?}", e);
                                JsError::throw(Kind::TypeError, e.description())
                            }
                        }
                    },
                    None => {
                        JsError::throw(Kind::TypeError, "unable to finish()")
                    }
                }
            });

            let id_map = result?;
            let mut buffer = JsArrayBuffer::new(scope, (id_map.len() * std::mem::size_of::<u32>()) as u32)?;
            buffer.grab(|mut data| {
                // ick ick ick -- there's no way to view this buffer as u32 in neon 0.1, so
                // this nastiness is necessary until we upgrade
                let slice = unsafe {
                    let ptr = std::mem::transmute::<*mut u8, *mut u32>(data.as_mut_ptr());
                    std::slice::from_raw_parts_mut(ptr, id_map.len())
                };
                slice.copy_from_slice(id_map.as_slice());
            });
            Ok(buffer.upcast())
        }
    }

    pub class JsFuzzyPhraseSet as JsFuzzyPhraseSet for FuzzyPhraseSet {
        init(mut call) {
            let filepath = call
                .check_argument::<JsString>(0)
                ?.value();
            match FuzzyPhraseSet::from_path(filepath) {
                Ok(set) => {
                    Ok(set)
                },
                Err(e) => {
                    println!("{:?}", e);
                    JsError::throw(Kind::TypeError, e.description())
                }
            }
        }

        method contains(call) {
            let phrase_array = call.arguments.require(call.scope, 0)?.check::<JsArray>()?;

            let arg1 = call.arguments.require(call.scope, 1)?;
            let ending_type: EndingType = neon_serde::from_value(
                call.scope,
                arg1
            )?;

            let mut v: Vec<String> = Vec::new();

            for i in 0..phrase_array.len() {
                let string = phrase_array.get(call.scope, i)
                ?.check::<JsString>()
                ?.value();

                v.push(string);
            }

            let mut this: Handle<JsFuzzyPhraseSet> = call.arguments.this(call.scope);

            let result = this.grab(|set| {
                match set.contains(v.as_slice(), ending_type) {
                    Ok(response) => {
                        Ok(response)
                    },
                    Err(e) => {
                        println!("{:?}", e);
                        JsError::throw(Kind::TypeError, e.description())
                    }
                }
            });

            Ok(JsBoolean::new(
                call.scope,
                result?
            ).upcast())
        }

        method fuzzyMatch(call) {
            let phrase_array = call.arguments.require(call.scope, 0)?.check::<JsArray>()?;
            let max_word_dist: u8 = call.arguments.require(call.scope, 1)?.check::<JsInteger>()
                ?.value() as u8;
            let max_phrase_dist: u8 = call.arguments.require(call.scope, 2)?.check::<JsInteger>()
                ?.value() as u8;

            let arg3 = call.arguments.require(call.scope, 3)?;
            let ending_type: EndingType = neon_serde::from_value(
                call.scope,
                arg3
            )?;

            let mut v: Vec<String> = Vec::new();

            for i in 0..phrase_array.len() {
                let string = phrase_array.get(call.scope, i)
                ?.check::<JsString>()
                ?.value();

                v.push(string);
            }

            let mut this: Handle<JsFuzzyPhraseSet> = call.arguments.this(call.scope);

            let result = this.grab(|set| {
                set.fuzzy_match(v.as_slice(), max_word_dist, max_phrase_dist, ending_type)
            });

            match result {
                Ok(vec) => {
                    let array = neon_serde::to_value(call.scope, &vec)?;

                    Ok(array.upcast())
                },
                Err(e) => {
                    println!("{:?}", e);
                    JsError::throw(Kind::TypeError, e.description())
                }
            }
        }

        method fuzzyMatchWindows(call) {
            let phrase_array = call.arguments.require(call.scope, 0)?.check::<JsArray>()?;
            let max_word_dist: u8 = call.arguments.require(call.scope, 1)?.check::<JsInteger>()
                ?.value() as u8;
            let max_phrase_dist: u8 = call.arguments.require(call.scope, 2)?.check::<JsInteger>()
                ?.value() as u8;

            let arg3 = call.arguments.require(call.scope, 3)?;
            let ending_type: EndingType = neon_serde::from_value(
                call.scope,
                arg3
            )?;

            let mut v: Vec<String> = Vec::new();

            for i in 0..phrase_array.len() {
                let string = phrase_array.get(call.scope, i)
                ?.check::<JsString>()
                ?.value();

                v.push(string);
            }

            let mut this: Handle<JsFuzzyPhraseSet> = call.arguments.this(call.scope);

            let result = this.grab(|set| {
                set.fuzzy_match_windows(v.as_slice(), max_word_dist, max_phrase_dist, ending_type)
            });

            match result {
                Ok(vec) => {
                    let array = neon_serde::to_value(call.scope, &vec)?;

                    Ok(array.upcast())
                },
                Err(e) => {
                    println!("{:?}", e);
                    JsError::throw(Kind::TypeError, e.description())
                }
            }
        }

        method fuzzyMatchMulti(call) {
            let arg0 = call.arguments.require(call.scope, 0)?;
            let multi_array: Vec<(Vec<String>, EndingType)> = neon_serde::from_value(
                call.scope,
                arg0
            )?;

            let max_word_dist: u8 = call.arguments.require(call.scope, 1)?.check::<JsInteger>()
                ?.value() as u8;
            let max_phrase_dist: u8 = call.arguments.require(call.scope, 2)?.check::<JsInteger>()
                ?.value() as u8;

            let mut this: Handle<JsFuzzyPhraseSet> = call.arguments.this(call.scope);

            let result = this.grab(|set| {
                set.fuzzy_match_multi(multi_array.as_slice(), max_word_dist, max_phrase_dist)
            });

            match result {
                Ok(vec) => {
                    let array = neon_serde::to_value(call.scope, &vec)?;

                    Ok(array.upcast())
                },
                Err(e) => {
                    println!("{:?}", e);
                    JsError::throw(Kind::TypeError, e.description())
                }
            }
        }

        method getByPhraseId(call) {
            let arg0 = call.arguments.require(call.scope, 0)?;
            let phrase_id: u32 = neon_serde::from_value(call.scope, arg0)?;

            let mut this: Handle<JsFuzzyPhraseSet> = call.arguments.this(call.scope);

            let result = this.grab(|set| {
                set.get_by_phrase_id(phrase_id)
            });

            match result {
                Ok(Some(vec)) => Ok(neon_serde::to_value(call.scope, &vec)?.upcast()),
                Ok(None) => Ok(JsUndefined::new().upcast()),
                Err(e) => JsError::throw(Kind::TypeError, e.description())
            }
        }

        method getPrefixBins(call) {
            let arg0 = call.arguments.require(call.scope, 0)?;
            let max_bin_size: usize = neon_serde::from_value(call.scope, arg0)?;

            let mut this: Handle<JsFuzzyPhraseSet> = call.arguments.this(call.scope);

            let result = this.grab(|set| {
                set.get_prefix_bins(max_bin_size)
            });

            match result {
                Ok(bins) => {
                    let mut bare_ids: Vec<u32> = bins.iter().map(|bin| bin.first.value() as u32).collect();
                    if let Some(bin) = bins.last() {
                        bare_ids.push(bin.last.value() as u32 + 1);
                    }

                    let mut buffer = JsArrayBuffer::new(call.scope, (bare_ids.len() * std::mem::size_of::<u32>()) as u32)?;
                    buffer.grab(|mut data| {
                        // again, there's no way to view this buffer as u32 in neon 0.1
                        let slice = unsafe {
                            let ptr = std::mem::transmute::<*mut u8, *mut u32>(data.as_mut_ptr());
                            std::slice::from_raw_parts_mut(ptr, bare_ids.len())
                        };
                        slice.copy_from_slice(bare_ids.as_slice());
                    });
                    Ok(buffer.upcast())
                },
                Err(e) => JsError::throw(Kind::TypeError, e.description())
            }
        }
    }
}

register_module!(m, {

    let class: Handle<JsClass<JsFuzzyPhraseSetBuilder>> = try!(JsFuzzyPhraseSetBuilder::class(m.scope));
    let constructor: Handle<JsFunction<JsFuzzyPhraseSetBuilder>> = try!(class.constructor(m.scope));
    try!(m.exports.set("FuzzyPhraseSetBuilder", constructor));

    let class: Handle<JsClass<JsFuzzyPhraseSet>> = try!(JsFuzzyPhraseSet::class(m.scope));
    let constructor: Handle<JsFunction<JsFuzzyPhraseSet>> = try!(class.constructor(m.scope));
    try!(m.exports.set("FuzzyPhraseSet", constructor));

    Ok(())
});
