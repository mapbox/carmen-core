const fuzzy = require('../lib');
const assert = require('assert');
const tape = require('tape');

tape('build FuzzyPhraseSetBuilder', (t) => {
    let builder = new fuzzy.FuzzyPhraseSetBuilder("set.fuzzy");
    t.ok(builder, "FuzzyPhraseSetBuilder built");
    t.throws(() => { new fuzzy.FuzzyPhraseSetBuilder()}, "throws on not enough arguments");
    t.throws(() => { new fuzzy.FuzzyPhraseSetBuilder("/etc/passwd")}, "throws on not a directory");
    t.throws(() => { new fuzzy.FuzzyPhraseSetBuilder({})}, "throws on wrong type argument");
    t.throws(() => { new fuzzy.FuzzyPhraseSetBuilder(7)}, "throws on wrong type arguments");
    t.end();
})

tape("FuzzyPhraseSetBuilder insertion and Set lookup", (t) => {
    let builder = new fuzzy.FuzzyPhraseSetBuilder("set.fuzzy");
    // ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]

    builder.insert(["100 main street"]);
    builder.insert(["200 main street"]);
    builder.insert(["100 main ave"]);
    builder.insert(["300 mlk blvd"]);
    builder.finish();

    let set = new fuzzy.FuzzyPhraseSet("set.fuzzy");
    t.ok(set.contains(["100 main street"]));
    t.ok(set.contains(["200 main street"]));
    t.ok(set.contains(["100 main ave"]));
    t.ok(set.contains(["300 mlk blvd"]));

    t.ok(set.contains(["100 main street"]));
    let phrase_static = ["100", "main", "street"];
    t.ok(set.contains(phrase_static));
    let phrase_vec = ["100", "main", "street"];
    t.ok(set.contains(phrase_vec));
    // let ref_phrase_vec = phrase_vec.iter().map(|s| s.as_str()).collect();
    // t.ok(set.contains(ref_phrase_vec));

    t.notOk(set.contains(["x"]));
    t.notOk(set.contains(["100 main"]));
    t.notOk(set.contains(["100 main s"]));
    t.notOk(set.contains(["100 main streetr"]));
    t.notOk(set.contains(["100 main street r"]));
    t.notOk(set.contains(["100 main street ave"]));

    t.ok(set.contains_prefix(["100 main street"]));
    t.ok(set.contains_prefix(["200 main street"]));
    t.ok(set.contains_prefix(["100 main ave"]));
    t.ok(set.contains_prefix(["300 mlk blvd"]));

    t.ok(set.contains_prefix(["100 main stree"]));
    t.ok(set.contains_prefix(["200 main stree"]));
    t.ok(set.contains_prefix(["100 main av"]));
    t.ok(set.contains_prefix(["300 mlk blv"]));

    t.ok(set.contains_prefix(["100 main"]));
    t.ok(set.contains_prefix(["200 main"]));
    t.ok(set.contains_prefix(["100 main"]));
    t.ok(set.contains_prefix(["300 mlk"]));

    t.ok(!set.contains_prefix(["100 man"]));
    t.ok(!set.contains_prefix(["400 main"]));
    t.ok(!set.contains_prefix(["100 main street x"]));

    // t.equals(
    //     set.fuzzy_match(["100", "man", "street"], 1, 1),
    //     { phrase: ["100", "main", "street"], edit_distance: 1 },
    // );
    //
    // t.equals(
    //     set.fuzzy_match(["100", "man", "stret"], 1, 2),
    //     { phrase: ["100", "main", "street"], edit_distance: 2 },
    // );
    //
    // t.equals(
    //     set.fuzzy_match_prefix(["100", "man"], 1, 1),
    //     { phrase: ["100", "main"], edit_distance: 1 },
    // );
    //
    // t.equals(
    //     set.fuzzy_match_prefix(["100", "man", "str"], 1, 1),
    //     { phrase: ["100", "main", "str"], edit_distance: 1 },
    // );


    t.end();
})
