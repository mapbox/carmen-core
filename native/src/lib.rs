#[macro_use]
extern crate neon;
extern crate fuzzy_phrase;

use std::path::Path;

use neon::mem::Handle;
use neon::vm::{This, Lock, FunctionCall, JsResult, Throw};
use neon::js::{JsFunction, Object, JsString, Value, JsUndefined, JsArray};
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
            let mut build = FuzzyPhraseSetBuilder::new(filename).unwrap();
            Ok(Some(build))
        }

        method insert(mut call) {
            let phrase = call
                .check_argument::<JsArray>(0);
                // ?.value();

            let scope = call.scope;
            let mut this: Handle<JsFuzzyPhraseSetBuilder> = call.arguments.this(scope);
                //loop over contents and check string
                // convert each to a rust string
                // place each in a vector that I have locally
                // once i have the rust vector then I'll pass that to the insert function
                // maybe weird moving from string => &str
            for word in phrase {
                word.check_argument::<JsString>(0);
                // ?.value();

                // let mut this: Handle<JsFuzzyPhraseSetBuilder> = call.arguments.this(scope);
                // this.grab(|fuzzyphrasesetbuilder| {
                //     match fuzzyphrasesetbuilder {
                //         Some(builder) => {
                //             builder.insert(&word).unwrap();
                //         },
                //         None => {
                //             panic!("FuzzyPhraseSetBuilder not available for insertion");
                //         }
                //     }
                // });
            };




            Ok(JsUndefined::new().upcast())
        }
    }

    // pub class JsFuzzyPhraseSet as JsFuzzyPhraseSet for Option<FuzzyPhraseSet<std::string::String>> {
    //     init(mut call) {
    //         let filename = call
    //             .check_argument::<JsString>(0)
    //             ?.value();
    //         let set = unsafe { FuzzyPhraseSet::from_path(filename).unwrap() };
    //         Ok(set)
    //     }
    // }
}

register_module!(m, {

    let class: Handle<JsClass<JsFuzzyPhraseSetBuilder>> = try!(JsFuzzyPhraseSetBuilder::class(m.scope));
    let constructor: Handle<JsFunction<JsFuzzyPhraseSetBuilder>> = try!(class.constructor(m.scope));
    try!(m.exports.set("FuzzyPhraseSetBuilder", constructor));

    Ok(())
});
