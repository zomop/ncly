//! A dataflow graph, its execution plan, and the effect of the optimiser.
//!
//! Run with: `cargo run --release -p ncly --example graph`

use std::collections::HashMap;

use ncly::compiler::{optimize_with_constants, Passes};
use ncly::graph::{Graph, Op};
use ncly::{NclyError, Tensor};

fn main() -> Result<(), NclyError> {
    // y = tanh(x·w + b) + tanh(x·w + b), a duplicate `tanh` added on purpose so
    // the optimiser has something to merge. `bias` and its `exp` are unreachable
    // from `y`, which gives dead-code elimination something to remove.
    let mut graph = Graph::new();
    let x = graph.input([4, 3], "x");
    let w = graph.input([3, 2], "w");
    let b = graph.input([2], "b");
    let bias = graph.input([2], "bias");
    let product = graph.add_node(Op::Matmul, &[x, w])?;
    let shifted = graph.add_node(Op::Add, &[product, b])?;
    let activation = graph.add_node(Op::Tanh, &[shifted])?;
    let duplicate = graph.add_node(Op::Tanh, &[shifted])?;
    let unused = graph.add_node(Op::Exp, &[bias])?;
    let y = graph.add_node(Op::Add, &[activation, duplicate])?;

    println!("{}", graph.describe());
    let schedule = graph.plan(&[y])?;
    println!(
        "schedule: {} steps, depth {}, {} dead",
        schedule.steps(),
        schedule.depth(),
        schedule.dead_nodes
    );
    for (level, nodes) in schedule.levels.iter().enumerate() {
        println!("  level {level}: {nodes:?}");
    }

    // Execute it.
    let mut values = HashMap::new();
    values.insert(
        x,
        Tensor::from_vec(
            vec![
                0.1, -0.4, 0.8, 0.5, 0.2, -0.9, -1.0, 0.3, 0.6, 0.0, 0.7, -0.2,
            ],
            [4, 3],
        )?,
    );
    values.insert(
        w,
        Tensor::from_vec(vec![0.3, -0.6, 0.9, 0.1, -0.2, 0.4], [3, 2])?,
    );
    values.insert(b, Tensor::from_vec(vec![0.05, -0.05], [2])?);
    values.insert(bias, Tensor::from_vec(vec![1.0, 2.0], [2])?);
    let execution = graph.run(&[y], &values)?;
    println!("y = {:?}", execution.get(y)?.to_vec());

    // Optimise. `bias` and its `exp` are dead; the two `tanh` nodes have the same
    // signature *and* the same input, so CSE collapses them into one. Note that
    // `w`, `b` and `bias` are three distinct leaves even though `b` and `bias`
    // share a shape: leaves carry caller-supplied values and are never merged.
    let mut constants = HashMap::new();
    constants.insert(b, values[&b].clone());
    let optimised = optimize_with_constants(&graph, &[y], &Passes::all(), &constants);
    println!(
        "optimised: {} -> {} nodes (dead {}, duplicates {}, folded {})",
        graph.len(),
        optimised.graph.len(),
        optimised.removed_dead,
        optimised.removed_duplicates,
        optimised.folded
    );
    println!("{}", optimised.graph.describe());

    // The optimised graph must produce the same answer.
    let mut reduced_values = HashMap::new();
    for (old, tensor) in [(x, &values[&x]), (w, &values[&w])] {
        if let Some(&new) = optimised.remap.get(&old.0) {
            reduced_values.insert(ncly::NodeId(new), tensor.clone());
        }
    }
    let new_output = optimised.remap_outputs(&[y])?[0];
    let reference = execution.get(y)?.to_vec();
    let rewritten = optimised.graph.run(&[new_output], &reduced_values)?;
    let again = rewritten.get(new_output)?.to_vec();
    assert_eq!(reference, again);
    println!("optimised graph reproduces the original result exactly.");
    assert_eq!(
        optimised.removed_duplicates, 1,
        "the duplicated tanh should be the only merge"
    );
    assert!(!optimised.remap.contains_key(&unused.0));
    Ok(())
}
