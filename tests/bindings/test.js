'use strict';

const addon = require('../../');
const tape = require('tape');
const tmp = require('tmp');
const rimraf = require('rimraf').sync;
const queue = require('d3-queue').queue;

tape('JsGridStoreBuilder init', (t) => {
    const tmpDir = tmp.dirSync();
    t.throws(() => new addon.GridStoreBuilder(), 'not enough arguments');
    t.throws(() => new addon.GridStoreBuilder({}), 'throws on wrong argument type');
    t.throws(() => new addon.GridStoreBuilder(7), 'throws on wrong argument type');
    const builder = new addon.GridStoreBuilder(tmpDir.name);
    t.ok(builder);
    rimraf(tmpDir.name);
    t.end();
});

tape('GridStoreBuilder insert()', (t) => {
    const tmpDir = tmp.dirSync();
    const builder = new addon.GridStoreBuilder(tmpDir.name);
    t.throws(() => builder.insert(), 'not enough arguments');
    t.throws(() => builder.insert({}), 'not enough arguments');
    builder.insert({ phrase_id: 0, lang_set: [0] }, [{ id: 0, x: 0, y: 0, relev: 0.5, score: 2, source_phrase_hash: 0 }]);
    builder.insert({ phrase_id: 1, lang_set: [0, 1, 2, 3] }, [{ id: 2, x: 2, y: 2, relev: 0.6, score: 3, source_phrase_hash: 0 }]);
    builder.finish();

    const reader = new addon.GridStore(tmpDir.name);
    t.deepEquals(reader.get({ phrase_id: 0, lang_set: [0] }), [ { relev: 1, score: 2, x: 0, y: 0, id: 0, source_phrase_hash: 0 } ], 'able to find the correct gridEntry inserted by insert()');
    t.deepEquals(reader.get({ phrase_id: 1, lang_set: [0, 1, 2, 3] }), [ { relev: 0.6, score: 3, x: 2, y: 2, id: 2, source_phrase_hash: 0 } ], 'able to find the correct gridEntry inserted by insert()');
    t.notOk(reader.get({ phrase_id: 3, lang_set: [3] }), 'cannot retrieve a grid that has not been inserted');
    t.end();
});

tape('GridStoreBuilder append()', (t) => {
    const tmpDir = tmp.dirSync();
    const builder = new addon.GridStoreBuilder(tmpDir.name);
    t.throws(() => builder.append(), 'not enough arguments');
    t.throws(() => builder.append({}), 'not enough arguments');
    builder.insert({ phrase_id: 0, lang_set: [0] }, [{ id: 0, x: 0, y: 0, relev: 0.5, score: 2, source_phrase_hash: 0 }]);
    builder.append({ phrase_id: 1, lang_set: [0, 2] }, [{ id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 }]);
    builder.finish();

    const reader = new addon.GridStore(tmpDir.name);
    let list = Array.from(reader.keys());
    t.deepEquals(list, [ { phrase_id: 0, lang_set: [ 0 ] }, { phrase_id: 1, lang_set: [ 0, 2 ] } ], 'GridStore contains the key inserted and the key appended');
    t.end();
});

tape('GridStoreBuilder compactAppend()', (t) => {
    const tmpDir = tmp.dirSync();
    const builder = new addon.GridStoreBuilder(tmpDir.name);
    builder.insert({ phrase_id: 0, lang_set: [0] }, [{ id: 0, x: 0, y: 0, relev: 1, score: 1, source_phrase_hash: 0 }]);
    builder.compactAppend({ phrase_id: 0, lang_set: [0] }, 1, 1, 0, 0, [[1, 1]]);
    builder.finish();

    const reader = new addon.GridStore(tmpDir.name);
    const list = Array.from(reader.keys());
    t.deepEquals(list, [{ phrase_id: 0, lang_set: [0] }], 'appended coordinates to the same feature');
    t.end();
});

tape('GridStoreBuilder finish()', (t) => {
    const tmpDir = tmp.dirSync();
    const builder = new addon.GridStoreBuilder(tmpDir.name);
    builder.insert({ phrase_id: 0, lang_set: [0] }, [{ id: 0, x: 0, y: 0, relev: 0.5, score: 2, source_phrase_hash: 0 }]);
    builder.insert({ phrase_id: 1, lang_set: [0, 1, 2, 3] }, [{ id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 }]);
    t.throws(() => new addon.GridStore(tmpDir.name), 'throws if you attempt to read without calling finish()');

    builder.finish();
    const reader = new addon.GridStore(tmpDir.name);
    t.ok(reader, 'can read only after finish() is called');
    t.end();
});

tape('GridStore reader', (t) => {
    const tmpDir = tmp.dirSync();
    const builder = new addon.GridStoreBuilder(tmpDir.name);
    builder.insert({ phrase_id: 0, lang_set: [0] }, [{ id: 0, x: 0, y: 0, relev: 0.5, score: 2, source_phrase_hash: 0 }]);
    builder.insert({ phrase_id: 1, lang_set: [0, 1, 2, 3] }, [{ id: 2, x: 2, y: 2, relev: 0.8, score: 3, source_phrase_hash: 0 }]);
    builder.finish();

    const reader = new addon.GridStore(tmpDir.name);
    let list = Array.from(reader.keys());
    t.deepEquals(list, [ { phrase_id: 0, lang_set: [ 0 ] }, { phrase_id: 1, lang_set: [ 0, 1, 2, 3 ] } ], 'GridStore is able to retrieve keys, reader works as expected');
    t.end();
});

tape('Coalesce tests - invalid inputs', (t) => {
    t.throws(() => {
        addon.coalesce();
    }, 'throws, incorrect arguments');

    t.throws(() => {
        addon.coalesce([]);
    }, 'throws, incorrect arguments');

    t.throws(() => {
        addon.coalesce([], {} );
    }, 'throws, incorrect arguments');

    t.throws(() => {
        addon.coalesce([{}], {}, () => {} );
    }, 'throws, incorrect arguments');

    t.throws(() => {
        addon.coalesce([-1], {}, () => {} );
    }, 'throws, incorrect argument type');

    t.throws(() => {
        addon.coalesce(undefined, {}, () => {} );
    }, 'throws, incorrect argument type');

    t.throws(() => {
        addon.coalesce([], {}, () => {} );
    }, 'throws, incorrect argument type');

    t.throws(() => {
        addon.coalesce([undefined], {}, () => {} );
    }, 'throws, incorrect argument type');

    t.throws(() => {
        addon.coalesce([null], {}, () => {} );
    }, 'throws, incorrect argument type');

    const tmpDir = tmp.dirSync();
    const builder = new addon.GridStoreBuilder(tmpDir.name);
    builder.insert({ phrase_id: 1, lang_set: [1] }, [{ id: 1, x: 2, y: 2, relev: 1., score: 1, source_phrase_hash: 0 }]);
    builder.finish();

    const reader = new addon.GridStore(tmpDir.name);

    // no weight
    const no_weight = [{
        store: reader,
		non_overlapping_indexes: [],
        match_key: {
            lang_set: [0],
            match_phrase: {
                "Range": {
                    start: 1,
                    end: 2
                }
            }
        },
        idx: 2,
        zoom: 6,
        mask: 1 << 1,
    }];
    t.throws(() => {addon.coalesce(no_weight, {}, () => {})}, 'no weight assigned in stack');

    // invalid type for gridstore
    const no_store = [{
        store: 'x',
        weight: 0.5,
        match_key: {
            lang_set: [0],
            match_phrase: {
                "Range": {
                    start: 1,
                    end: 2
                }
            }
        },
        idx: 2,
        zoom: 6,
        mask: 1 << 1,
    }];
    t.throws(() => {addon.coalesce(no_store, {}, () => {})}, /failed downcast to JsGridStore/, 'invalid stack');

    const no_match_key = [{
        store: reader,
        non_overlapping_indexes: [],
        weight: 0.5,
        idx: 2,
        zoom: 6,
        mask: 1 << 1,
    }];
    t.throws(() => {addon.coalesce(no_match_key, {}, () => {})}, 'no match_key');

    const no_idx = [{
        store: reader,
        non_overlapping_indexes: [],
        weight: 0.5,
        match_key: {
            lang_set: [0],
            match_phrase: {
                "Range": {
                    start: 1,
                    end: 2
                }
            }
        },
        zoom: 6,
        mask: 1 << 1,
    }];
    t.throws(() => {addon.coalesce(no_idx, {}, () => {})}, 'no idx');

    const no_mask = [{
        store: reader,
        non_overlapping_indexes: [],
        weight: 0.5,
        match_key: {
            lang_set: [0],
            match_phrase: {
                "Range": {
                    start: 1,
                    end: 2
                }
            }
        },
        zoom: 6,
    }];
    t.throws(() => {addon.coalesce(no_idx, {}, () => {})}, 'no mask');
    t.end();
});

tape('Coalesce single valid stack - Valid inputs', (t) => {
    const tmpDir = tmp.dirSync();
    const builder = new addon.GridStoreBuilder(tmpDir.name);
    builder.insert({ phrase_id: 1, lang_set: [1] },
        [
            { id: 1, x: 2, y: 2, relev: 1., score: 1, source_phrase_hash: 0 },
            { id: 2, x: 2, y: 0, relev: 1., score: 1, source_phrase_hash: 0 },
            { id: 3, x: 0, y: 0, relev: 1., score: 1, source_phrase_hash: 0 }
        ]
    );
    builder.finish();
    const readerOpts = { idx: 1, zoom: 14, non_overlapping_indexes: Array.from(new Set()), type_id: 0, coalesce_radius: 200, bboxes: globalBboxForZoom(14) };
	const reader = new addon.GridStore(tmpDir.name, readerOpts);

    const valid_stack = [{
        store: reader,
        non_overlapping_indexes: [],
        weight: 1,
        match_key: {
            lang_set: [1],
            match_phrase: {
                "Range": {
                    start: 1,
                    end: 3
                }
            }
        },
        idx: 1,
        zoom: 14,
        mask: 1 << 0,
        id: 0
    }];

    addon.coalesce(valid_stack, { zoom: 14 }, (err, res) => {
        t.deepEqual(res[0].entries,
            [
                { grid_entry:
                    { relev: 1, score: 1, x: 2, y: 2, id: 1, source_phrase_hash: 0 },
                    matches_language: true,
                    idx: 1,
                    tmp_id: 33554433,
                    mask: 1,
                    distance: 0,
                    scoredist: 1,
                    phrasematch_id: 0,
                }
            ], 'Ok, finds the right grid entry');
        t.equal(res.length, 3, 'Result set has 3 grid entries');
        t.end();
    });
});

tape('Coalesce multi valid stack - Valid inputs', (t) => {
    const tmpDir1 = tmp.dirSync();
    const builder1 = new addon.GridStoreBuilder(tmpDir1.name);
    builder1.insert({ phrase_id: 1, lang_set: [1] },
        [
            { id: 1, x: 1, y: 1, relev: 1., score: 1, source_phrase_hash: 0 }
        ]
    );
    builder1.finish();
    const reader1Opts = { idx: 0, zoom: 1, non_overlapping_indexes: Array.from(new Set()), type_id: 0, coalesce_radius: 200, bboxes: globalBboxForZoom(1) };
	const reader1 = new addon.GridStore(tmpDir1.name, reader1Opts);

    const tmpDir2 = tmp.dirSync();
    const builder2 = new addon.GridStoreBuilder(tmpDir2.name);
    builder2.insert({ phrase_id: 2, lang_set: [1] },
        [
            { id: 1, x: 1, y: 1, relev: 1., score: 3, source_phrase_hash: 0 },
            { id: 2, x: 2, y: 2, relev: 1., score: 3, source_phrase_hash: 0 }
        ]
    );
    builder2.finish();
    const reader2Opts = { idx: 1, zoom: 2, non_overlapping_indexes: Array.from(new Set()), type_id: 1, coalesce_radius: 200, bboxes: globalBboxForZoom(2) };
	const reader2 = new addon.GridStore(tmpDir2.name, reader2Opts);

    const valid_coalesce_multi = [{
        store: reader1,
		non_overlapping_indexes: reader1Opts.non_overlapping_indexes,
        weight: 0.5,
        match_key: {
            lang_set: [1],
            match_phrase: {
                "Range": {
                    start: 1,
                    end: 3
                }
            }
        },
        idx: 0,
        zoom: 1,
        mask: 1 << 1,
        id: 0
    },{
        store: reader2,
		non_overlapping_indexes: reader2Opts.non_overlapping_indexes,
        weight: 0.5,
        match_key: {
            lang_set: [1],
            match_phrase: {
                "Range": {
                    start: 1,
                    end: 3
                }
            }
        },
        idx: 1,
        zoom: 2,
        mask: 1 << 0,
        id: 1
    }];
    addon.coalesce(valid_coalesce_multi, { zoom: 2 }, (err, res) => {
        t.deepEqual(res[0].entries[0].grid_entry, { relev: 0.5, score: 3, x: 2, y: 2, id: 2, source_phrase_hash: 0 }, '1st result highest score from the higher zoom index');
        t.deepEqual(res[0].entries.length, 2, 'Result set has 2 grid entries');
        t.end();
    });
});

tape('lang_set >= 128', (t) => {
    const tmpDir = tmp.dirSync();
    const builder = new addon.GridStoreBuilder(tmpDir.name);
    builder.insert({ phrase_id: 1, lang_set: [1] },
        [
            { id: 1, x: 2, y: 2, relev: 1., score: 1, source_phrase_hash: 0 }
        ]
    );
    builder.finish();
    const reader = new addon.GridStore(tmpDir.name);

    const lang_set_stack = [{
        store: reader,
		non_overlapping_indexes: [],
        weight: 1,
        match_key: {
            lang_set: [128],
            match_phrase: {
                "Range": {
                    start: 1,
                    end: 3
                }
            }
        },
        idx: 1,
        zoom: 14,
        mask: 1 << 0,
        id: 0
    }];

    addon.coalesce(lang_set_stack, { zoom: 14 }, (err, res) => {
        t.deepEqual(res[0].entries[0].matches_language, false, 'does not match language - ignore lang_set > 128');
        t.deepEqual(res[0].relev, 0.96, 'relevance penalised - ignore lang_set > 128');
    });
    t.end();
});

tape('Bin boundaries', (t) => {
    const directoryWithBoundaries = tmp.dirSync();
    const directoryWithoutBoundaries = tmp.dirSync();

    const builderWithBoundaries = new addon.GridStoreBuilder(directoryWithBoundaries.name);
    const builderWithoutBoundaries = new addon.GridStoreBuilder(directoryWithoutBoundaries.name);

    // this will produce 5000 phrases aaa, aab, aac, ...
    const alphabet = 'abcdefghijklmnopqrstuvwxyz';
    const phrases = [];

    outermost:
    for (const l1 of alphabet) {
        for (const l2 of alphabet) {
            for (const l3 of alphabet) {
                phrases.push(l1 + l2 + l3);
                if (phrases.length >= 5000) break outermost;
            }
        }
    }

    // insert phrases
    for (let i = 0; i < phrases.length; i++) {
        let key = { phrase_id: i, lang_set: [1] };
        let entries = [{
            id: i,
            x: i,
            y: 1,
            relev: 1.,
            score: 1,
            source_phrase_hash: 0,
        }];
        builderWithBoundaries.insert(key, entries);
        builderWithoutBoundaries.insert(key, entries);
    }

    // calculate bins
    let bins = new Map();
    for (let i = 0; i < phrases.length; i++) {
        // insert the first occurrence of every prefix
        if (!bins.has(phrases[i].charAt(0))) {
            bins.set(phrases[i].charAt(0), i);
        }
    }
    const boundaries = Array.from(bins.values());
    boundaries.push(phrases.length);
    boundaries.sort((a, b) => a - b);

    builderWithBoundaries.loadBinBoundaries(Uint32Array.from(boundaries).buffer);

    builderWithBoundaries.finish();
    builderWithoutBoundaries.finish();

    const readerWithBoundariesOpts = { idx: 1, zoom: 14, non_overlapping_indexes: Array.from(new Set()), type_id: 0, coalesce_radius: 200, bboxes: globalBboxForZoom(14) };
	const readerWithBoundaries = new addon.GridStore(directoryWithBoundaries.name, readerWithBoundariesOpts);
    const readerWithoutBoundariesOpts = { idx: 1, zoom: 14, non_overlapping_indexes: Array.from(new Set()), type_id: 0, coalesce_radius: 200, bboxes: globalBboxForZoom(14) };
	const readerWithoutBoundaries = new addon.GridStore(directoryWithoutBoundaries.name, readerWithoutBoundariesOpts);

    const findRange = (prefix) => {
        let start = null,
            end = null;
        for (let i = 0; i < phrases.length; i++) {
            if (phrases[i].startsWith(prefix)) {
                if (start === null) start = i;
                end = i;
            }
        }
        return { start, end };
    };

    let startsWithB = findRange('b');
    let startsWithBc = findRange('bc');

    // try via coalesce, comparing the two backends
    const q = queue();
    [
        { reader: readerWithBoundaries, range: startsWithB },
        { reader: readerWithoutBoundaries, range: startsWithB },
        { reader: readerWithBoundaries, range: startsWithBc },
        { reader: readerWithoutBoundaries, range: startsWithBc }
    ].forEach(({ reader, range }) => {
        const subquery = {
            store: reader,
		    non_overlapping_indexes: [],
            weight: 1.,
            match_key: { match_phrase: { "Range": range }, lang_set: [1] },
            idx: 1,
            zoom: 14,
            mask: 1,
            id: 0
        };
        let stack = [subquery];
        let match_opts = {
            zoom: 14
        };
        q.defer((cb) => addon.coalesce(stack, match_opts, cb));
    });

    q.awaitAll((err, results) => {
        t.deepEquals(results[0], results[1], 'things that start with b are the same with and without bins');
        t.deepEquals(results[2], results[3], 'things that start with bc are the same with and without bins');
        t.end();
    });
});

tape('Deserialize phrasematch results', (t) => {

    const tmpDir = tmp.dirSync();
    const builder = new addon.GridStoreBuilder(tmpDir.name);
    builder.insert({ phrase_id: 1, lang_set: [1] },
        [
            { id: 1, x: 1, y: 1, relev: 1., score: 1, source_phrase_hash: 0 }
        ]
    );
    builder.finish();
    const storeOpts = { idx: 0, zoom: 14, non_overlapping_indexes: Array.from(new Set()), type_id: 0, coalesce_radius: 200, bboxes: globalBboxForZoom(14) };
	const store = new addon.GridStore(tmpDir.name, storeOpts);
    let phrasematchResults = [
        new Phrasematch(store, ['main', 'street'], 'main street', 1, [0, 2], 1, 0, 14, 6, 1, false, false, false, 0, ['main', 'street'], 0, 14, [0], 1, 0, [])
    ];
    addon.stackable(phrasematchResults);
    t.end();
});

function globalBboxForZoom(zoom) {
    let max = (1 << zoom) - 1;
    return [[0, 0, max, max]];
}

function Phrasematch(store, subquery, phrase, weight, phrase_id_range, scorefactor, prefix, mask, zoom, editMultiplier, prox_match, cat_match, partial_number, subquery_edit_distance, original_phrase, original_phrase_ender, original_phrase_mask, languages, id, idx, non_overlapping_indexes) {
    this.store = store;
    this.subquery = subquery;
    this.phrase = phrase;
    this.weight = weight;
    this.mask = mask;
    this.phrase_id_range = phrase_id_range;
    this.scorefactor = scorefactor;
    this.prefix = prefix;
    this.zoom = zoom;
    this.editMultiplier = editMultiplier || 1;
    this.prox_match = prox_match || false;
    this.cat_match = cat_match || false;
    this.partial_number = partial_number || false;
    this.subquery_edit_distance = subquery_edit_distance;
    this.original_phrase = original_phrase;
    this.original_phrase_ender = original_phrase_ender;
    this.original_phrase_mask = original_phrase_mask;
    this.id = id;
    this.idx = idx;
    this.non_overlapping_indexes = [];

    if (languages) {
        // carmen-cache gives special treatment to the "languages" property
        // being absent, so if we don't get one passed in, don't pass it through
        this.languages = languages;
    }

    // format a couple of the items the way carmen-core expects them
    this.match_key = {
        lang_set: this.languages,
        match_phrase: {
            'Range': {
                start: this.phrase_id_range[0],
                end: this.phrase_id_range[1] + 1
            }
        }
    };
}

module.exports.Phrasematch = Phrasematch;
