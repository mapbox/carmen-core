#[macro_use]
extern crate neon;
extern crate fuzzy_phrase;

use neon::mem::Handle;
use neon::vm::{This, Lock, FunctionCall, JsResult};
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
            let phrase_array = call
                .check_argument::<JsArray>(0)?;

            // let &mut scope = call.scope;

            let mut v: Vec<String> = Vec::new();

            for i in 0..phrase_array.len() {
                let string = phrase_array.get(call.scope, i)
                ?.check::<JsString>()
                ?.value();

                v.push(string);
            }

            let mut this: Handle<JsFuzzyPhraseSetBuilder> = call.arguments.this(call.scope);
                //loop over contents and check string
                // convert each to a rust string
                // place each in a vector that I have locally
                // once i have the rust vector then I'll pass that to the insert function
                // maybe weird moving from string => &str

            // for word in phrase {

                // let mut this: Handle<JsFuzzyPhraseSetBuilder> = call.arguments.this(scope);

                // convert string => &str
                // let string_word = format!(word);
                // let immutable_word = &word;

                // place word in a vector
            this.grab(|fuzzyphrasesetbuilder| {
                match fuzzyphrasesetbuilder {
                    Some(builder) => {
                        // once referencing the vector, insert the word
                        let mut v2: Vec<&str> = v.into_iter().map(
                            |el| el.as_str()
                        ).collect();

                        builder.insert(&v2[..]).unwrap();
                    },
                    None => {
                        panic!("FuzzyPhraseSetBuilder not available for insertion");
                    }
                }
            });
            // };
            Ok(JsUndefined::new().upcast())
        }

        // method insert_str() {
        //
        // }

        // method finish(mut call) {
        //     let scope = call.scope;
        //         let mut this: Handle<JsSetBuilder> = call.arguments.this(scope);
        //
        //         this.grab(|setbuilder| {
        //             match setbuilder.take() {
        //                 Some(builder) => {
        //                     builder.finish();
        //                 },
        //                 None => {
        //                     panic!("SetBuilder not available for finish()");
        //                 }
        //             }
        //         });
        //     Ok(JsUndefined::new().upcast())
        // }
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
