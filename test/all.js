'use strict';

const tape = require('tape');
const tmp = require('tmp');
const rimraf = require('rimraf').sync;
const fs = require('fs');
const fuzzy = require('../lib');

tape('build FuzzyPhraseSetBuilder', (t) => {
    const tmpDir = tmp.dirSync();

    const builder = new fuzzy.FuzzyPhraseSetBuilder(tmpDir.name);
    t.ok(builder, 'FuzzyPhraseSetBuilder built');
    t.throws(() => {
        new fuzzy.FuzzyPhraseSetBuilder();
    }, 'throws on not enough arguments');
    t.throws(() => {
        new fuzzy.FuzzyPhraseSetBuilder('/etc/passwd');
    }, 'throws on not a directory');
    t.throws(() => {
        new fuzzy.FuzzyPhraseSetBuilder({});
    }, 'throws on wrong type argument');
    t.throws(() => {
        new fuzzy.FuzzyPhraseSetBuilder(7);
    }, 'throws on wrong type arguments');

    rimraf(tmpDir.name);

    t.end();
});

tape('FuzzyPhraseSetBuilder insertion and Set lookup', (t) => {
    const tmpDir = tmp.dirSync();

    const builder = new fuzzy.FuzzyPhraseSetBuilder(tmpDir.name);

    builder.insert(['100', 'main', 'street']);
    builder.insert(['200', 'main', 'street']);
    builder.insert(['100', 'main', 'ave']);
    builder.insert(['300', 'mlk', 'blvd']);
    builder.finish();

    const set = new fuzzy.FuzzyPhraseSet(tmpDir.name);
    t.ok(set.contains(['100', 'main', 'street'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');
    t.ok(set.contains(['200', 'main', 'street'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');
    t.ok(set.contains(['100', 'main', 'ave'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');
    t.ok(set.contains(['300', 'mlk', 'blvd'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');

    t.ok(set.contains(['100', 'main', 'street'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');
    const phraseStatic = ['100', 'main', 'street'];
    t.ok(set.contains(phraseStatic, fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');
    const phraseVec = ['100', 'main', 'street'];
    t.ok(set.contains(phraseVec, fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet contains()');

    t.notOk(set.contains(['x'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet does not contains()');
    t.notOk(set.contains(['100', 'main'], fuzzy.ENDING_TYPE.nonPrefix), 'FuzzyPhraseSet does not contains()');
    t.notOk(
        set.contains(['100', 'main', 's'], fuzzy.ENDING_TYPE.nonPrefix),
        'FuzzyPhraseSet does not contains()'
    );
    t.notOk(
        set.contains(['100', 'main', 'streetr'], fuzzy.ENDING_TYPE.nonPrefix),
        'FuzzyPhraseSet does not contains()'
    );
    t.notOk(
        set.contains(['100', 'main', 'street', 'r'], fuzzy.ENDING_TYPE.nonPrefix),
        'FuzzyPhraseSet does not contains()'
    );
    t.notOk(
        set.contains(['100', 'main', 'street', 'ave'], fuzzy.ENDING_TYPE.nonPrefix),
        'FuzzyPhraseSet does not contains()'
    );

    t.ok(
        set.contains(['100', 'main', 'street'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet containsPrefix()'
    );
    t.ok(
        set.contains(['200', 'main', 'street'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet containsPrefix()'
    );
    t.ok(
        set.contains(['100', 'main', 'ave'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet containsPrefix()'
    );
    t.ok(
        set.contains(['300', 'mlk', 'blvd'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet containsPrefix()'
    );

    t.ok(set.contains(['100', 'main'], fuzzy.ENDING_TYPE.anyPrefix), 'FuzzyPhraseSet containsPrefix()');
    t.ok(set.contains(['200', 'main'], fuzzy.ENDING_TYPE.anyPrefix), 'FuzzyPhraseSet containsPrefix()');
    t.ok(set.contains(['100', 'main'], fuzzy.ENDING_TYPE.anyPrefix), 'FuzzyPhraseSet containsPrefix()');
    t.ok(set.contains(['300', 'mlk'], fuzzy.ENDING_TYPE.anyPrefix), 'FuzzyPhraseSet containsPrefix()');

    t.notOk(
        set.contains(['100', 'man'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet does not containsPrefix()'
    );
    t.notOk(
        set.contains(['400', 'main'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet does not containsPrefix()'
    );
    t.notOk(
        set.contains(['100', 'main', 'street', 'x'], fuzzy.ENDING_TYPE.anyPrefix),
        'FuzzyPhraseSet does not containsPrefix()'
    );

    t.ok(
        set.fuzzyMatch(['100', 'man', 'street'], 1, 1, fuzzy.ENDING_TYPE.nonPrefix),
        'FuzzyPhraseSet fuzzyMatch(..., fuzzy.ENDING_TYPE.nonPrefix)'
    );

    t.deepEquals(
        set.fuzzyMatch(['100', 'man', 'street'], 1, 1, fuzzy.ENDING_TYPE.nonPrefix),
        [{ edit_distance: 1, phrase: ['100', 'main', 'street'], ending_type: fuzzy.ENDING_TYPE.nonPrefix, phrase_id_range: [ 1, 1 ] }],
        'FuzzyPhraseSet fuzzyMatch(..., fuzzy.ENDING_TYPE.nonPrefix)'
    );

    t.deepEquals(
        set.fuzzyMatch(['100', 'man', 'stret'], 1, 2, fuzzy.ENDING_TYPE.nonPrefix),
        [{ edit_distance: 2, phrase: ['100', 'main', 'street'], ending_type: fuzzy.ENDING_TYPE.nonPrefix, phrase_id_range: [ 1, 1 ] }],
        'FuzzyPhraseSet fuzzyMatch(..., fuzzy.ENDING_TYPE.nonPrefix)'
    );

    t.deepEquals(
        set.fuzzyMatch(['100', 'man'], 1, 1, fuzzy.ENDING_TYPE.anyPrefix),
        [{ phrase: ['100', 'main'], edit_distance: 1, ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix, phrase_id_range: [ 0, 1 ] }],
        'FuzzyPhraseSet fuzzyMatch(..., fuzzy.ENDING_TYPE.anyPrefix)'
    );

    t.deepEquals(
        set.fuzzyMatch(['100', 'man', 'str'], 1, 1, fuzzy.ENDING_TYPE.anyPrefix),
        [{ phrase: ['100', 'main', 'str'], edit_distance: 1, ending_type: fuzzy.ENDING_TYPE.anyPrefix, phrase_id_range: [ 1, 1 ] }],
        'FuzzyPhraseSet fuzzyMatch(..., fuzzy.ENDING_TYPE.anyPrefix)'
    );

    t.deepEquals(
        set.fuzzyMatchMulti(
            [
                [['100', 'man', 'street'], fuzzy.ENDING_TYPE.nonPrefix],
                [['100', 'man', 'stret'], fuzzy.ENDING_TYPE.nonPrefix],
                [['100', 'man'], fuzzy.ENDING_TYPE.anyPrefix],
                [['100', 'man', 'str'], fuzzy.ENDING_TYPE.anyPrefix]
            ],
            1,
            2
        ),
        [
            [{ edit_distance: 1, phrase: ['100', 'main', 'street'], ending_type: fuzzy.ENDING_TYPE.nonPrefix, phrase_id_range: [ 1, 1 ] }],
            [{ edit_distance: 2, phrase: ['100', 'main', 'street'], ending_type: fuzzy.ENDING_TYPE.nonPrefix, phrase_id_range: [ 1, 1 ] }],
            [{ phrase: ['100', 'main'], edit_distance: 1, ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix, phrase_id_range: [ 0, 1 ] }],
            [{ phrase: ['100', 'main', 'str'], edit_distance: 1, ending_type: fuzzy.ENDING_TYPE.anyPrefix, phrase_id_range: [ 1, 1 ] }]
        ],
        'FuzzyPhraseSet fuzzyMatchMulti()'
    );

    t.deepEquals(
        set.fuzzyMatchWindows(
            '100 man street washington 200'.split(' '),
            0,
            0,
            fuzzy.ENDING_TYPE.nonPrefix
        ),
        []
    );

    t.deepEquals(
        set.fuzzyMatchWindows(
            '100 man street washington 200'.split(' '),
            1,
            1,
            fuzzy.ENDING_TYPE.nonPrefix
        ),
        [
            {
                edit_distance: 1,
                ending_type: fuzzy.ENDING_TYPE.nonPrefix,
                phrase: ['100', 'main', 'street'],
                start_position: 0,
                phrase_id_range: [ 1, 1 ]
            }
        ]
    );

    t.deepEquals(
        set.fuzzyMatchWindows(
            '100 man street washington 200'.split(' '),
            1,
            1,
            fuzzy.ENDING_TYPE.anyPrefix
        ),
        [
            {
                edit_distance: 1,
                ending_type: fuzzy.ENDING_TYPE.nonPrefix,
                phrase: ['100', 'main', 'street'],
                start_position: 0,
                phrase_id_range: [ 1, 1 ]
            },
            {
                edit_distance: 0,
                ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix,
                phrase: ['200'],
                start_position: 4,
                phrase_id_range: [ 2, 2 ]
            }
        ]
    );

    rimraf(tmpDir.name);

    t.end();
});

tape('word replacements', (t) => {
    const tmpDir = tmp.dirSync();
    const replacements = [
        { from: "street", to: "st" },
        { from: "saint", to: "st" },
        { from: "avenue", to: "ave" },
        { from: "fort", to: "ft" },
        { from: "road", to: "rd" },
    ];

    const builder = new fuzzy.FuzzyPhraseSetBuilder(tmpDir.name);
    builder.loadWordReplacements(replacements);

    builder.insert(["100", "main", "street"]);
    builder.insert(["100", "main", "st"]);
    builder.insert(["100", "maine", "st"]);
    builder.insert(["100", "ft", "wayne", "rd"]);
    builder.insert(["100", "fortenberry", "ave"]);

    builder.finish();

    const word_replacements_obj = JSON.parse(fs.readFileSync(tmpDir.name + '/metadata.json')).word_replacements;
    t.deepEquals(word_replacements_obj, replacements, 'ok, loads word replacements');

    const set = new fuzzy.FuzzyPhraseSet(tmpDir.name);

    // regular fuzzy search
    t.deepEquals(
        set.fuzzyMatch("100 main s".split(' '), 1, 1, fuzzy.ENDING_TYPE.anyPrefix),
        [
            { edit_distance: 0, phrase: ["100", "main", "s"], ending_type: fuzzy.ENDING_TYPE.anyPrefix, phrase_id_range: [ 2, 2 ] },
            { edit_distance: 1, phrase: ["100", "maine", "s"], ending_type: fuzzy.ENDING_TYPE.anyPrefix, phrase_id_range: [ 3, 3 ] }
        ]
    );

    t.deepEquals(
        set.fuzzyMatch("100 main st".split(' '), 1, 1, fuzzy.ENDING_TYPE.anyPrefix),
        [
            { edit_distance: 0, phrase: ["100", "main", "st"], ending_type: fuzzy.ENDING_TYPE.anyPrefix, phrase_id_range: [ 2, 2 ] },
            { edit_distance: 1, phrase: ["100", "maine", "st"], ending_type: fuzzy.ENDING_TYPE.anyPrefix, phrase_id_range: [ 3, 3 ] }
        ]
    );

    t.deepEquals(
        set.fuzzyMatch("100 main str".split(' '), 1, 1, fuzzy.ENDING_TYPE.anyPrefix),
        [
            { edit_distance: 0, phrase: ["100", "main", "st"], ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix, phrase_id_range: [ 2, 2 ] },
            { edit_distance: 1, phrase: ["100", "maine", "st"], ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix, phrase_id_range: [ 3, 3 ] }
        ]
    );

    t.deepEquals(
        set.fuzzyMatch("100 main str".split(' '), 0, 0, fuzzy.ENDING_TYPE.anyPrefix),
        [
            { edit_distance: 0, phrase: ["100", "main", "st"], ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix, phrase_id_range: [ 2, 2 ] },
        ]
    );

    t.deepEquals(
        set.fuzzyMatch("100 main stre".split(' '), 1, 1, fuzzy.ENDING_TYPE.anyPrefix),
        [
            { edit_distance: 0, phrase: ["100", "main", "st"], ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix, phrase_id_range: [ 2, 2 ] },
            { edit_distance: 1, phrase: ["100", "maine", "st"], ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix, phrase_id_range: [ 3, 3 ] }
        ]
    );

    t.deepEquals(
        set.fuzzyMatch("100 main stree".split(' '), 1, 1, fuzzy.ENDING_TYPE.anyPrefix),
        [
            { edit_distance: 0, phrase: ["100", "main", "st"], ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix, phrase_id_range: [ 2, 2 ] },
            { edit_distance: 1, phrase: ["100", "maine", "st"], ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix, phrase_id_range: [ 3, 3 ] }
        ]
    );

    t.deepEquals(
        set.fuzzyMatch("100 main street".split(' '), 1, 1, fuzzy.ENDING_TYPE.anyPrefix),
        [
            { edit_distance: 0, phrase: ["100", "main", "st"], ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix, phrase_id_range: [ 2, 2 ] },
            { edit_distance: 1, phrase: ["100", "maine", "st"], ending_type: fuzzy.ENDING_TYPE.wordBoundaryPrefix, phrase_id_range: [ 3, 3 ] }
        ]
    );

    t.deepEquals(
        set.fuzzyMatch("100 main s".split(' '), 1, 1, fuzzy.ENDING_TYPE.nonPrefix),
        []
    );

    t.deepEquals(
        set.fuzzyMatch("100 main st".split(' '), 1, 1, fuzzy.ENDING_TYPE.nonPrefix),
        [
            { edit_distance: 0, phrase: ["100", "main", "st"], ending_type: fuzzy.ENDING_TYPE.nonPrefix, phrase_id_range: [ 2, 2 ] },
            { edit_distance: 1, phrase: ["100", "maine", "st"], ending_type: fuzzy.ENDING_TYPE.nonPrefix, phrase_id_range: [ 3, 3 ] }
        ]
    );

    // windowed search
    const variants = [
        "100 fort wayne road washington dc",
        "100 ft wayne road washington dc",
        "100 fort wayne rd washington dc",
        "100 ft wayne rd washington dc"
    ];
    for (const variant of variants) {
        t.deepEquals(
            set.fuzzyMatchWindows(variant.split(' '), 1, 1, fuzzy.ENDING_TYPE.anyPrefix),
            [
                {
                    edit_distance: 0,
                    phrase: ["100", "ft", "wayne", "rd"],
                    start_position: 0,
                    ending_type: fuzzy.ENDING_TYPE.nonPrefix,
                    phrase_id_range: [1, 1]
                }
            ]
        )
    }

    // multi search -- just test it in terms of regular fuzzy search, since we know that works
    const stuffToTry = [
        ["100", "main", "s"],
        ["100", "main", "st"],
        ["100", "main", "str"],
        ["100", "main", "stre"],
        ["100", "main", "stree"],
        ["100", "main", "street"],
        ["100", "fort", "wayne", "road"],
        ["100", "ft", "wayne", "road"],
        ["100", "fort", "wayne", "rd"],
        ["100", "ft", "wayne", "rd"]
    ];
    const regularResults = [];
    const multiToTry = [];
    for (const ending of [fuzzy.ENDING_TYPE.nonPrefix, fuzzy.ENDING_TYPE.anyPrefix, fuzzy.ENDING_TYPE.wordBoundaryPrefix]) {
        for (const phrase of stuffToTry) {
            regularResults.push(set.fuzzyMatch(phrase, 1, 1, ending));
            multiToTry.push([phrase, ending]);
        }
    }
    t.deepEquals(regularResults, set.fuzzyMatchMulti(multiToTry, 1, 1));

    rimraf(tmpDir.name);

    t.end();
});
