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

    builder.insert(["100","main","street"]);
    builder.insert(["200","main","street"]);
    builder.insert(["100","main","ave"]);
    builder.insert(["300","mlk","blvd"]);
    builder.finish();

    let set = new fuzzy.FuzzyPhraseSet("set.fuzzy");
    t.ok(set.contains(["100","main","street"]), "FuzzyPhraseSet contains()");
    t.ok(set.contains(["200","main","street"]), "FuzzyPhraseSet contains()");
    t.ok(set.contains(["100","main","ave"]), "FuzzyPhraseSet contains()");
    t.ok(set.contains(["300","mlk","blvd"]));

    t.ok(set.contains(["100","main","street"]), "FuzzyPhraseSet contains()");
    let phrase_static = ["100", "main", "street"];
    t.ok(set.contains(phrase_static), "FuzzyPhraseSet contains()");
    let phrase_vec = ["100", "main", "street"];
    t.ok(set.contains(phrase_vec), "FuzzyPhraseSet contains()");

    t.notOk(set.contains(["x"]), "FuzzyPhraseSet does not contains()");
    t.notOk(set.contains(["100","main"]), "FuzzyPhraseSet does not contains()");
    t.notOk(set.contains(["100","main","s"]), "FuzzyPhraseSet does not contains()");
    t.notOk(set.contains(["100","main","streetr"]), "FuzzyPhraseSet does not contains()");
    t.notOk(set.contains(["100","main","street","r"]), "FuzzyPhraseSet does not contains()");
    t.notOk(set.contains(["100","main","street","ave"]), "FuzzyPhraseSet does not contains()");

    t.ok(set.contains_prefix(["100","main","street"]), "FuzzyPhraseSet contains_prefix()");
    t.ok(set.contains_prefix(["200","main","street"]), "FuzzyPhraseSet contains_prefix()");
    t.ok(set.contains_prefix(["100","main","ave"]), "FuzzyPhraseSet contains_prefix()");
    t.ok(set.contains_prefix(["300","mlk","blvd"]), "FuzzyPhraseSet contains_prefix()");

    t.ok(set.contains_prefix(["100","main"]), "FuzzyPhraseSet contains_prefix()");
    t.ok(set.contains_prefix(["200","main"]), "FuzzyPhraseSet contains_prefix()");
    t.ok(set.contains_prefix(["100","main"]), "FuzzyPhraseSet contains_prefix()");
    t.ok(set.contains_prefix(["300","mlk"]), "FuzzyPhraseSet contains_prefix()");

    t.notOk(set.contains_prefix(["100","man"]), "FuzzyPhraseSet does not contains_prefix()");
    t.notOk(set.contains_prefix(["400","main"]), "FuzzyPhraseSet does not contains_prefix()");
    t.notOk(set.contains_prefix(["100","main","street","x"]), "FuzzyPhraseSet does not contains_prefix()");

    t.ok(set.fuzzy_match(["100", "man", "street"], 1, 1), "FuzzyPhraseSet fuzzy_match()");

    t.deepEquals(
        set.fuzzy_match(["100", "man", "street"], 1, 1),
        [ { edit_distance: 1, phrase: [ '100', 'main', 'street' ] } ],
        "FuzzyPhraseSet fuzzy_match()"
    );

    t.deepEquals(
        set.fuzzy_match(["100", "man", "stret"], 1, 2),
        [ { edit_distance: 2, phrase: [ '100', 'main', 'street' ] } ],
        "FuzzyPhraseSet fuzzy_match()"
    );

    t.deepEquals(
        set.fuzzy_match_prefix(["100", "man"], 1, 1),
        [{ phrase: ["100", "main"], edit_distance: 1 }],
        "FuzzyPhraseSet fuzzy_match_prefix()"
    );

    t.deepEquals(
        set.fuzzy_match_prefix(["100", "man", "str"], 1, 1),
        [{ phrase: ["100", "main", "str"], edit_distance: 1 }],
        "FuzzyPhraseSet fuzzy_match_prefix()"
    );


    t.end();
})
