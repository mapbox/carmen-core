#[macro_use]
extern crate neon;
extern crate fuzzy_phrase;

use neon::vm::{Call, JsResult, FunctionCall, This};
use neon::js::{Value};
use neon::js::JsString;

use fuzzy_phrase::glue::{FuzzyPhraseSetBuilder, FuzzyPhraseSet};

// check argument type
trait CheckArgument {
  fn check_argument<V: Value>(&mut self, i: i32) -> JsResult<V>;
}

impl<'a, T: This> CheckArgument for FunctionCall<'a, T> {
  fn check_argument<V: Value>(&mut self, i: i32) -> JsResult<V> {
    self.arguments.require(self.scope, i)?.check::<V>()
  }
}

// FuzzyPhraseSetBuilder::new!(fuzzy_phrase::FuzzyPhraseSetBuilder);

declare_types! {
    pub class JsFuzzyPhraseSetBuilder as JsFuzzyPhraseSetBuilder for Option<FuzzyPhraseSetBuilder<std::string::String>> {
        init(mut call) {
            let path = call
                .check_argument::<JsString>(0)
                ?.value();
            Ok(())
        }
    }

    pub class JsFuzzyPhraseSet as JsFuzzyPhraseSet for Option<FuzzyPhraseSet<std::string::String>> {
        init(mut call) {

            Ok(())
        }
    }
}

register_module!(m, {


});
