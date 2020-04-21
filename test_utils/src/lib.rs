extern crate carmen_core;
extern crate failure;
extern crate serde;
extern crate serde_json;

use carmen_core::gridstore::*;

use failure::Error;
use lz4::Decoder;
use rusoto_core::Region;
use rusoto_s3::{GetObjectRequest, S3Client, S3};
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

// Util functions for tests and benchmarks

/// Round a float to a number of digits past the decimal point
pub fn round(value: f64, digits: i32) -> f64 {
    let multiplier = 10.0_f64.powi(digits);
    (value * multiplier).round() / multiplier
}

/// Convert an array of language ids into the langfield to use for GridKey or MatchKey
pub fn langarray_to_langfield(array: &[u32]) -> u128 {
    let mut out = 0u128;
    for lang in array {
        out = out | (1 << *lang as usize);
    }
    out
}

/// Mapping of GridKey to all of the grid entries to insert into a store for that GridKey
#[derive(Serialize, Deserialize, Debug)]
pub struct StoreEntryBuildingBlock {
    pub grid_key: GridKey,
    pub entries: Vec<GridEntry>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PrefixBoundary {
    prefix: String,
    first: u32,
    last: u32,
}

pub struct TestStore {
    pub store: GridStore,
    pub idx: u16,
    pub non_overlapping_indexes: HashSet<u16>,
}

/// Utility to create stores
/// Takes an vector, with each item mapping to a store to create
/// Each item is a vector with maps of grid keys to the entries to insert into the store for that grid key
pub fn create_store(
    store_entries: Vec<StoreEntryBuildingBlock>,
    idx: u16,
    zoom: u16,
    type_id: u16,
    non_overlapping_indexes: HashSet<u16>,
    coalesce_radius: f64,
) -> TestStore {
    let directory: tempfile::TempDir = tempfile::tempdir().unwrap();
    let mut builder = GridStoreBuilder::new(directory.path()).unwrap();
    for build_block in store_entries {
        builder.insert(&build_block.grid_key, build_block.entries).expect("Unable to insert");
    }
    builder.finish().unwrap();
    TestStore {
        store: GridStore::new_with_options(directory.path(), zoom, type_id, coalesce_radius)
            .unwrap(),
        idx,
        non_overlapping_indexes,
    }
}

// Gets the absolute path for a path relative to the carmen-core dir
pub fn get_absolute_path(relative_path: &Path) -> Result<PathBuf, Error> {
    let dir = env::current_dir().expect("Error getting current dir");
    let mut filepath = fs::canonicalize(&dir).expect("Error getting cannonicalized current dir");
    filepath.push(relative_path);
    Ok(filepath)
}

/// Load grid data from a local JSON path
pub fn load_db_from_json(json_path: &str, split_path: &str, store_path: &str) {
    // Open json file
    let json_path = Path::new(json_path);
    let json_f = File::open(json_path).expect("Error opening file");
    let json_file = io::BufReader::new(json_f);

    let split_path = Path::new(split_path);
    let split_f = File::open(split_path).expect("Error opening file");
    let split_file = io::BufReader::new(split_f);

    load_db_from_json_reader(json_file, split_file, store_path);
}

fn load_db_from_json_reader<T: BufRead>(json_source: T, split_source: T, store_path: &str) {
    // Set up new gridstore
    let directory = Path::new(store_path);
    let mut builder = GridStoreBuilder::new(directory).unwrap();
    json_source.lines().for_each(|l| {
        let record = l.unwrap();
        if !record.is_empty() {
            let deserialized: StoreEntryBuildingBlock =
                serde_json::from_str(&record).expect("Error deserializing json from string");
            builder.insert(&deserialized.grid_key, deserialized.entries).expect("Unable to insert");
        }
    });

    let boundaries: Vec<u32> =
        serde_json::from_reader(split_source).expect("Error deserializing json from string");
    builder.load_bin_boundaries(boundaries).unwrap();

    builder.finish().unwrap();
}

/// Takes an absolute path (in string form) to a rocksdb dir, and an absolute path for the output file,
/// reads the data from the db, and writes a json representation of the data to a file
pub fn dump_db_to_json(store_path: &str, json_path: &str) {
    let reader = GridStore::new(store_path).unwrap();
    let output_file = File::create(json_path).unwrap();
    let mut writer = BufWriter::new(output_file);
    for item in reader.iter() {
        let (grid_key, entries) = item.unwrap();
        let key_record_pair = StoreEntryBuildingBlock { grid_key, entries };
        let line = serde_json::to_string(&key_record_pair).expect("Unable to serialize record");
        let bytes = line.as_bytes();
        writer.write(&bytes).unwrap();
        writer.write(b"\n").unwrap();
    }

    let mut boundaries: Vec<u32> = reader.bin_boundaries.iter().cloned().collect();
    boundaries.sort();
    let splits_path = json_path.to_owned().replace(".gridstore.dat", "") + ".gridstore.splits";
    let splits_file = File::create(splits_path).unwrap();
    let mut splits_writer = BufWriter::new(splits_file);
    splits_writer.write(serde_json::to_string(&boundaries).unwrap().as_bytes()).unwrap();
}

pub fn ensure_downloaded(datafile: &str) -> PathBuf {
    let tmp = std::env::temp_dir().join("carmen_core_data/downloads");
    std::fs::create_dir_all(&tmp).unwrap();
    let path = tmp.join(Path::new(datafile));
    if !path.exists() {
        let client = S3Client::new(Region::UsEast1);
        let request = GetObjectRequest {
            bucket: "mapbox".to_owned(),
            key: ("playground/apendleton/gridstore_bench_v2/".to_owned() + datafile),
            ..Default::default()
        };

        let result = client.get_object(request).sync().unwrap();

        let stream = result.body.unwrap();
        let mut body: Vec<u8> = Vec::new();
        stream.into_blocking_read().read_to_end(&mut body).unwrap();

        let mut file = File::create(&path).expect("create failed");
        file.write_all(&body).expect("failed to write body");
    }

    path
}

pub const GRIDSTORE_DATA_SUFFIX: &'static str = ".gridstore.dat.lz4";
pub const PREFIX_BOUNDARY_SUFFIX: &'static str = ".gridstore.splits.lz4";

pub fn ensure_store(datafile: &str) -> PathBuf {
    let tmp = std::env::temp_dir().join("carmen_core_data/indexes");
    std::fs::create_dir_all(&tmp).unwrap();
    let idx_path = tmp.join(Path::new(&datafile.replace(".dat.lz4", ".rocksdb")));
    if !idx_path.exists() {
        let grid_path = ensure_downloaded(datafile);
        let splits_path =
            ensure_downloaded(&datafile.replace(GRIDSTORE_DATA_SUFFIX, PREFIX_BOUNDARY_SUFFIX));

        let grid_decoder = Decoder::new(File::open(grid_path).unwrap()).unwrap();
        let grid_file = io::BufReader::new(grid_decoder);

        let splits_decoder = Decoder::new(File::open(splits_path).unwrap()).unwrap();
        let splits_file = io::BufReader::new(splits_decoder);

        load_db_from_json_reader(grid_file, splits_file, idx_path.to_str().unwrap());
    }

    idx_path
}

#[derive(Deserialize, Debug)]
pub struct GridStorePlaceholder {
    path: String,
    zoom: u16,
    type_id: u16,
    coalesce_radius: f64,
}

#[derive(Deserialize, Debug)]
struct SubqueryPlaceholder {
    store: GridStorePlaceholder,
    idx: u16,
    non_overlapping_indexes: HashSet<u16>,
    weight: f64,
    match_keys: Vec<MatchKeyWithId>,
    mask: u32,
}

pub fn prepare_phrasematches(
    datafile: &str,
) -> Vec<(Vec<PhrasematchSubquery<Rc<GridStore>>>, MatchOpts)> {
    let path = ensure_downloaded(datafile);
    let decoder = Decoder::new(File::open(path).unwrap()).unwrap();
    let file = io::BufReader::new(decoder);
    let mut stores: HashMap<String, Rc<GridStore>> = HashMap::new();
    let out: Vec<(Vec<PhrasematchSubquery<Rc<GridStore>>>, MatchOpts)> = file
        .lines()
        .filter_map(|l| {
            let record = l.unwrap();
            if !record.is_empty() {
                let deserialized: (Vec<SubqueryPlaceholder>, MatchOpts) =
                    serde_json::from_str(&record).expect("Error deserializing json from string");
                let stack: Vec<_> = deserialized
                    .0
                    .iter()
                    .map(|placeholder| {
                        let store =
                            stores.entry(placeholder.store.path.clone()).or_insert_with(|| {
                                let store_name = placeholder
                                    .store
                                    .path
                                    .rsplit("/")
                                    .next()
                                    .unwrap()
                                    .replace(".rocksdb", ".dat.lz4");
                                let store_path = ensure_store(&store_name);
                                let gs = GridStore::new_with_options(
                                    store_path,
                                    placeholder.store.zoom,
                                    placeholder.store.type_id,
                                    placeholder.store.coalesce_radius,
                                )
                                .unwrap();
                                Rc::new(gs)
                            });
                        PhrasematchSubquery {
                            store: store.clone(),
                            weight: placeholder.weight,
                            match_keys: placeholder.match_keys.clone(),
                            mask: placeholder.mask,
                            idx: placeholder.idx,
                            non_overlapping_indexes: placeholder.non_overlapping_indexes.clone(),
                        }
                    })
                    .collect();

                Some((stack, deserialized.1))
            } else {
                None
            }
        })
        .collect();
    out
}

pub fn prepare_stackable_phrasematches(
    datafile: &str,
) -> Vec<Vec<PhrasematchSubquery<Rc<GridStore>>>> {
    let path = ensure_downloaded(datafile);
    let decoder = Decoder::new(File::open(path).unwrap()).unwrap();
    let file = io::BufReader::new(decoder);
    let mut stores: HashMap<String, Rc<GridStore>> = HashMap::new();
    let out: Vec<Vec<PhrasematchSubquery<Rc<GridStore>>>> = file
        .lines()
        .filter_map(|l| {
            let record = l.unwrap();
            if !record.is_empty() {
                let deserialized: (Vec<SubqueryPlaceholder>, MatchOpts) =
                    serde_json::from_str(&record).expect("Error deserializing json from string");
                let stack: Vec<_> = deserialized
                    .0
                    .iter()
                    .map(|placeholder| {
                        let store =
                            stores.entry(placeholder.store.path.clone()).or_insert_with(|| {
                                // since stackable doesn't really need the actual gridstore data
                                // we're using aa-country in order to avoid having to download gridstore data from every index
                                let store_name =
                                    "aa-country-both-3e43d23805-069d003ff2.gridstore.dat.lz4";
                                let store_path = ensure_store(&store_name);
                                let gs = GridStore::new_with_options(
                                    store_path,
                                    placeholder.store.zoom,
                                    placeholder.store.type_id,
                                    placeholder.store.coalesce_radius,
                                )
                                .unwrap();
                                Rc::new(gs)
                            });
                        PhrasematchSubquery {
                            store: store.clone(),
                            weight: placeholder.weight,
                            match_keys: placeholder.match_keys.clone(),
                            mask: placeholder.mask,
                            idx: placeholder.idx,
                            non_overlapping_indexes: placeholder.non_overlapping_indexes.clone(),
                        }
                    })
                    .collect();
                Some(stack)
            } else {
                None
            }
        })
        .collect();
    out
}
