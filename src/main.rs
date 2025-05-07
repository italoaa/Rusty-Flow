use rflow::optimizers::SGD;
use rflow::tensor::{Tensor, TensorRef};

// polars
use polars::prelude::*;
use std::fs::File;
//rand
use indicatif::ProgressBar;
use std::io::Write;

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
    fn new(dataset: DataFrame, batch_size: usize, shuffle: bool) -> Self {
        let mut batches = Vec::new();

        // get the number of rows
        let num_rows = dataset.height();

        // progress bar
        let pb = ProgressBar::new(6400);

        // create batches
        for i in (0..6400).step_by(batch_size) {
            let end = std::cmp::min(i + batch_size, num_rows);
            let mut batch_df = dataset.slice(i as i64, end - i as usize);
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

                    pixels.push(value / 255.0); // Normalize pixel values to [0, 1]
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

            // Update the progress bar
            pb.inc(1);
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

    let batch_size = 64;
    let shuffle = true;
    let data_loader = DataLoader::new(dataset.train, batch_size, shuffle);

    println!(
        "DataLoader created with {} batches of size {}",
        data_loader.batches.len(),
        batch_size
    );

    // Parameters
    // The NN has a hidden layer of 128 and an output layer of 10
    // if the input is batchsize, 784
    // the hidden layer is batchsize, 128
    // the weights are 784, 128
    let w1 = Tensor::kaiming_init(vec![784, 128], 0.1);
    let b1 = Tensor::zeros_like(vec![1, 128]);

    let w2 = Tensor::kaiming_init(vec![128, 64], 0.1);
    let b2 = Tensor::zeros_like(vec![1, 64]);

    let w3 = Tensor::kaiming_init(vec![64, 10], 0.1);
    let b3 = Tensor::zeros_like(vec![1, 10]);

    let params = vec![w1.rc(), b1.rc(), w2.rc(), b2.rc(), w3.rc(), b3.rc()];
    let mut optim = SGD::new(params, 0.001, 0.0, 0.0);

    // Epoch bar
    for epoch in 0..10 {
        let mut losses = vec![];
        for (i, batch) in data_loader.batches.iter().enumerate() {
            optim.zero_grad();
            // One hot
            let labels = batch.labels.one_hot(10);

            // Forward
            let xw1 = batch.data.mm(&w1);
            let preacts1 = &xw1 + &b1;
            let acts1 = preacts1.lrelu(0.1);
            let xw2 = acts1.mm(&w2);
            let preacts2 = &xw2 + &b2;
            let acts2 = preacts2.lrelu(0.1);
            let xw3 = acts2.mm(&w3);
            let logits = &xw3 + &b3;

            // Loss calculation
            let cross_entropy = logits.cross_entropy_with_logits(&labels);
            let loss = cross_entropy.mean(0);

            // backward pass and update
            loss.backward();

            // update the weights
            optim.step();

            // Print loss and carridge return
            losses.push(loss.data.borrow()[0]);
            print!(
                "\rEpoch: {} Sample: {}, Loss: {:?}",
                epoch,
                i,
                loss.data.borrow()[0]
            );

            // force a flush
            std::io::stdout().flush().unwrap();
        }

        let avg_loss: f32 = losses.iter().sum::<f32>() / losses.len() as f32;

        // Print average loss
        println!("\nEpoch: {} Average Loss: {:?}", epoch, avg_loss);
    }
}
