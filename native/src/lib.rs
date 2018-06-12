#[macro_use]
extern crate neon;
extern crate fuzzy_phrase;

use neon::mem::Handle;
use neon::vm::{This, Lock, FunctionCall, JsResult};
use neon::js::{JsFunction, Object, JsString, Value, JsUndefined, JsArray, JsBoolean};
use neon::js::class::{JsClass, Class};

use fuzzy_phrase::glue::{FuzzyPhraseSetBuilder, FuzzyPhraseSet};

// check argument type: https://github.com/Brooooooklyn/sourcemap-decoder/blob/master/native/src/lib.rs#L21-L29
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
            let build = FuzzyPhraseSetBuilder::new(filename).unwrap();
            Ok(Some(build))
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

            this.grab(|fuzzyphrasesetbuilder| {
                match fuzzyphrasesetbuilder {
                    Some(builder) => {
                        builder.insert(&v[..]).unwrap();
                    },
                    None => {
                        panic!("FuzzyPhraseSetBuilder not available for insertion");
                    }
                };
            });

            Ok(JsUndefined::new().upcast())
        }

        method finish(call) {
            let scope = call.scope;
                let mut this: Handle<JsFuzzyPhraseSetBuilder> = call.arguments.this(scope);

                this.grab(|fuzzyphrasesetbuilder| {
                    match fuzzyphrasesetbuilder.take() {
                        Some(builder) => {
                            builder.finish();
                        },
                        None => {
                            panic!("SetBuilder not available for finish()");
                        }
                    }
                });
            Ok(JsUndefined::new().upcast())
        }
    }

    pub class JsFuzzyPhraseSet as JsFuzzyPhraseSet for FuzzyPhraseSet {
        init(mut call) {
            let filename = call
                .check_argument::<JsString>(0)
                ?.value();
            let set = { FuzzyPhraseSet::from_path(filename).unwrap() };
            Ok(set)
        }

        method from_path(call) {
            // the directory path of the set with all subcomponents at predictable URLS
            let path_array = call.arguments.require(call.scope, 0)?.check::<JsArray>()?;

            let mut v: Vec<String> = Vec::new();

            for i in 0..path_array.len() {
                let string = path_array.get(call.scope, i)
                ?.check::<JsString>()
                ?.value();

                v.push(string);
            }
            Ok(JsUndefined::new().upcast())
        }

        // method contains(call) {
        //     let word = call
        //         .check_argument::<JsString>(0)
        //         ?.value();
        //     let scope = call.scope;
        //     let mut this: Handle<JsFuzzyPhraseSet> = call.arguments.this(scope);
        //
        //     Ok(JsBoolean::new(
        //         scope,
        //         this.grab(|set| {
        //             set.contains(&word).unwrap()
        //         })
        //     ).upcast())
        // }
        //
        // method contains_prefix() {
        //
        // }
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
