use rflow::tensor::{Tensor, TensorRef};

// polars
use polars::prelude::*;
use std::fs::File;
//rand

struct Batch {
    data: TensorRef,
    labels: TensorRef,
}

struct Dataset {
    train: DataFrame,
    test: DataFrame,
}

struct DataLoader {
    batches: Vec<Batch>,
    batch_size: usize,
    shuffle: bool,
}

impl Dataset {
    fn new(train: File, test: File) -> Self {
        let train_df = CsvReader::new(train).finish().unwrap();
        let test_df = CsvReader::new(test).finish().unwrap();

        Dataset {
            train: train_df,
            test: test_df,
        }
    }
}

impl DataLoader {
    fn new(dataset: Dataset, batch_size: usize, shuffle: bool) -> Self {
        let mut batches = Vec::new();
        let train_df = dataset.train;
        let _test_df = dataset.test;

        // get the number of rows
        let num_rows = train_df.height();

        // create batches
        for i in (0..10).step_by(batch_size) {
            let end = std::cmp::min(i + batch_size, num_rows);
            let mut batch_df = train_df.slice(i as i64, end - i as usize);
            let labels: Vec<i64> = batch_df
                .column("label")
                .unwrap()
                .i64()
                .unwrap()
                .into_iter()
                .map(|opt| opt.unwrap())
                .collect();

            // turn labels into f32
            let labels: Vec<f32> = labels.iter().map(|&x| x as f32).collect();
            let labels: TensorRef =
                Tensor::new_with_options(labels, vec![batch_size, 1], false, None, vec![]);

            // Drop the label column to work with just pixel data
            batch_df.drop_in_place("label").unwrap();

            // Convert the DataFrame to a 2D array of f32
            // If the data is
            // 1, 2, 3
            // 4, 5, 6
            // we want to convert it to: 1, 2, 3, 4, 5, 6
            // Convert the DataFrame to a 2D array of f32
            let mut pixels: Vec<f32> = Vec::with_capacity(batch_df.height() * batch_df.width());

            // Get all column names
            let column_names = batch_df.get_column_names();

            // Iterate through each row and column to flatten the DataFrame
            for row_idx in 0..batch_df.height() {
                for col_name in column_names.iter() {
                    // Get the value at the current position and convert to f32
                    let value = batch_df
                        .column(col_name)
                        .unwrap()
                        .get(row_idx)
                        .unwrap()
                        .try_extract::<f64>()
                        .unwrap_or(0.0) as f32;

                    pixels.push(value);
                }
            }

            // Create a TensorRef from the flattened pixel data
            let pixels: TensorRef =
                Tensor::new_with_options(pixels, vec![batch_size, 784], false, None, vec![]);

            // Create a batch
            let batch = Batch {
                data: pixels,
                labels,
            };

            // Add the batch to the list of batches

            batches.push(batch);
        }

        // shuffle the batches if needed
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::rng();
            batches.shuffle(&mut rng);
        }

        DataLoader {
            batches,
            batch_size,
            shuffle,
        }
    }
}

fn main() {
    let mnist = "../mnist";
    // let train_csv = format!("{}/mnist_train.csv", mnist);
    let train_file = File::open(format!("{}/mnist_train.csv", mnist)).unwrap();
    // let test_csv = format!("{}/mnist_test.csv", mnist);
    let test_file = File::open(format!("{}/mnist_test.csv", mnist)).unwrap();

    let dataset = Dataset::new(train_file, test_file);

    let batch_size = 2;
    let shuffle = true;
    let data_loader = DataLoader::new(dataset, batch_size, shuffle);

    println!(
        "ba {}, suf: {}",
        data_loader.batch_size, data_loader.shuffle
    );

    println!(
        "DataLoader created with {} batches of size {}",
        data_loader.batches.len(),
        batch_size
    );

    let i = 0;
    for batch in &data_loader.batches {
        println!("Batch data shape: {:?}", batch.data.shape);
        println!("Batch labels shape: {:?}", batch.labels.shape);

        // THe NN has a hidden layer of 128 and an output layer of 10
        // if the input is batchsize, 784
        // the hidden layer is batchsize, 128
        // the weights are 784, 128
        let w1 = Tensor::new_random(vec![784, 128]);
        let b1 = Tensor::ones_like(vec![128]);

        let w2 = Tensor::new_random(vec![128, 10]);
        let b2 = Tensor::ones_like(vec![10]);

        // forward pass
        // xw1 = x * w1
        // acts = xw + b1
        let xw1 = batch.data.mm(&w1);
        let acts1 = &xw1 + &b1;
        acts1.relu();

        let xw2 = acts1.mm(&w2);
        let logits = &xw2 + &b2;
        let cross_entropy = logits.cross_entropy(&batch.labels);
        let loss = cross_entropy.mean();
        println!("Batch {}: Loss: {:?}", i, loss);
        loss.backward();
    }

    println!("Hello, world!");
}
