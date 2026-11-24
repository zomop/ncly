//! Graph and compiler validation against the direct tensor API.
//!
//! The point of these tests is *independence*: the expected values are computed
//! by calling `ncly-tensor` directly, not by re-running the graph. A bug shared
//! between the executor and its own expected values would not be caught, so the
//! reference path deliberately shares no code with `ncly-graph`.

use std::collections::HashMap;

use ncly_compiler::{optimize, optimize_with_constants, Passes};
use ncly_graph::{Graph, NodeId, Op};
use ncly_integration_tests::assert_close;
use ncly_tensor::Tensor;

/// `tanh(exp(a) * b + a)` computed with the tensor API alone.
fn reference(a: &Tensor, b: &Tensor) -> Vec<f64> {
    a.exp().mul(b).unwrap().add(a).unwrap().tanh().to_vec()
}

/// Build `tanh(exp(a) * b + a)` as a graph.
fn build(graph: &mut Graph) -> (NodeId, NodeId, NodeId) {
    let a = graph.input([3, 4], "a");
    let b = graph.input([3, 4], "b");
    let exp = graph.add_node(Op::Exp, &[a]).unwrap();
    let product = graph.add_node(Op::Mul, &[exp, b]).unwrap();
    let shifted = graph.add_node(Op::Add, &[product, a]).unwrap();
    let output = graph.add_node(Op::Tanh, &[shifted]).unwrap();
    (a, b, output)
}

/// The constant chain used by the folding tests: `tanh(exp(t))` applied twice,
/// then negated. Every step is total on the whole real line apart from `exp`
/// overflow, which `Tensor::uniform` values cannot trigger.
const CHAIN: [Op; 4] = [Op::Exp, Op::Tanh, Op::Neg, Op::Exp];

/// Apply [`CHAIN`] with the tensor API, one node at a time.
fn reference_chain(input: &Tensor) -> Vec<f64> {
    let mut value = input.clone();
    for op in CHAIN {
        value = match op {
            Op::Exp => value.exp(),
            Op::Tanh => value.tanh(),
            Op::Neg => value.neg(),
            _ => unreachable!("CHAIN only uses unary ops defined here"),
        };
    }
    value.to_vec()
}

#[test]
fn graph_execution_matches_the_direct_tensor_api() {
    let mut graph = Graph::new();
    let (a, b, output) = build(&mut graph);

    let a_value = Tensor::uniform([3, 4], 41);
    let b_value = Tensor::uniform([3, 4], 42);
    let mut values = HashMap::new();
    values.insert(a, a_value.clone());
    values.insert(b, b_value.clone());

    let execution = graph.run(&[output], &values).unwrap();
    let from_graph = execution.get(output).unwrap().to_vec();
    assert_close(
        &from_graph,
        &reference(&a_value, &b_value),
        1e-15,
        "graph execution vs the tensor API",
    );
}

#[test]
fn optimised_graph_agrees_with_the_unoptimised_one() {
    let mut graph = Graph::new();
    let (a, b, output) = build(&mut graph);

    let a_value = Tensor::uniform([3, 4], 43);
    let b_value = Tensor::uniform([3, 4], 44);

    // `b` is a compile-time constant, but `a` is not, so nothing can fold. That is
    // the case worth testing: an optimiser must not "helpfully" evaluate a
    // partially-known expression.
    let mut constants = HashMap::new();
    constants.insert(b, b_value.clone());
    let optimised = optimize_with_constants(&graph, &[output], &Passes::all(), &constants);
    assert_eq!(optimised.folded, 0, "{}", optimised.graph.describe());

    let mut values = HashMap::new();
    values.insert(a, a_value.clone());
    values.insert(b, b_value.clone());
    let plain = graph
        .run(&[output], &values)
        .unwrap()
        .get(output)
        .unwrap()
        .to_vec();

    let target = NodeId(optimised.remap[&output.0]);
    let mut remapped = HashMap::new();
    remapped.insert(NodeId(optimised.remap[&a.0]), a_value);
    let rewritten = optimised
        .graph
        .run(&[target], &remapped)
        .unwrap()
        .get(target)
        .unwrap()
        .to_vec();

    assert_eq!(
        plain, rewritten,
        "optimisation must be invisible in the result"
    );
}

#[test]
fn fully_constant_subgraph_is_baked_into_a_leaf() {
    let mut graph = Graph::new();
    let x = graph.input([2, 2], "x");
    let constant = graph.input([2, 2], "c");
    let mut tail = constant;
    for op in CHAIN {
        tail = graph.add_node(op, &[tail]).unwrap();
    }
    let output = graph.add_node(Op::Add, &[tail, x]).unwrap();

    let c_value = Tensor::uniform([2, 2], 45);
    let x_value = Tensor::uniform([2, 2], 46);
    let mut constants = HashMap::new();
    constants.insert(constant, c_value.clone());

    let optimised = optimize_with_constants(&graph, &[output], &Passes::all(), &constants);
    assert_eq!(
        optimised.folded,
        CHAIN.len(),
        "every constant op should fold: {}",
        optimised.graph.describe()
    );

    // Only `x` is supplied: the chain is baked in and `c` is gone.
    let mut values = HashMap::new();
    values.insert(NodeId(optimised.remap[&x.0]), x_value);
    let target = NodeId(optimised.remap[&output.0]);
    let actual = optimised
        .graph
        .run(&[target], &values)
        .expect("the optimised graph must not need the constant")
        .get(target)
        .unwrap()
        .to_vec();

    let expected: Vec<f64> = reference_chain(&c_value)
        .iter()
        .zip(x_value.to_vec())
        .map(|(folded, extra)| folded + extra)
        .collect();
    assert_close(
        &actual,
        &expected,
        1e-14,
        "baked constant vs the tensor API",
    );
}

#[test]
fn scheduling_is_deterministic_and_respects_dependencies() {
    let mut graph = Graph::new();
    let (_, _, output) = build(&mut graph);
    let first = graph.plan(&[output]).unwrap();
    let second = graph.plan(&[output]).unwrap();
    assert_eq!(
        first.order, second.order,
        "scheduling must be deterministic"
    );

    let mut position = HashMap::new();
    for (index, id) in first.order.iter().enumerate() {
        position.insert(*id, index);
    }
    for id in &first.order {
        for parent in &graph.node(*id).unwrap().inputs {
            assert!(
                position[parent] < position[id],
                "node {id} was scheduled before its input {parent}"
            );
        }
    }
}

#[test]
fn folding_never_reassociates_floating_point_arithmetic() {
    // 0.1 + 0.2 + 0.3 differs from 0.1 + (0.2 + 0.3) in binary floating point.
    // Folding must reproduce the *source* order exactly, not a "simplified" one.
    let mut graph = Graph::new();
    let a = graph.input([1], "a");
    let b = graph.input([1], "b");
    let c = graph.input([1], "c");
    let ab = graph.add_node(Op::Add, &[a, b]).unwrap();
    let abc = graph.add_node(Op::Add, &[ab, c]).unwrap();

    let mut constants = HashMap::new();
    constants.insert(a, Tensor::from_vec(vec![0.1], [1]).unwrap());
    constants.insert(b, Tensor::from_vec(vec![0.2], [1]).unwrap());
    constants.insert(c, Tensor::from_vec(vec![0.3], [1]).unwrap());

    let optimised = optimize_with_constants(&graph, &[abc], &Passes::all(), &constants);
    let target = NodeId(optimised.remap[&abc.0]);
    let baked = optimised
        .graph
        .run(&[target], &HashMap::new())
        .unwrap()
        .get(target)
        .unwrap()
        .to_vec();

    let left_to_right = 0.1 + 0.2 + 0.3;
    let regrouped = 0.1 + (0.2 + 0.3);
    assert_ne!(
        left_to_right, regrouped,
        "the test is only meaningful if the groupings differ"
    );
    assert_eq!(
        baked,
        vec![left_to_right],
        "folding must preserve evaluation order"
    );
}

#[test]
fn dead_code_elimination_does_not_change_the_result() {
    let mut graph = Graph::new();
    let (a, b, output) = build(&mut graph);
    // A large unused branch.
    let junk = graph.add_node(Op::Exp, &[a]).unwrap();
    let junk = graph.add_node(Op::Exp, &[junk]).unwrap();
    let junk = graph.add_node(Op::Neg, &[junk]).unwrap();
    assert_eq!(graph.len(), 4 + 3, "leaves, main path and the dead branch");

    let optimised = optimize(&graph, &[output], &Passes::all());
    assert_eq!(optimised.removed_dead, 3, "{}", optimised.graph.describe());

    let a_value = Tensor::uniform([3, 4], 47);
    let b_value = Tensor::uniform([3, 4], 48);
    let mut values = HashMap::new();
    values.insert(a, a_value.clone());
    values.insert(b, b_value.clone());
    let plain = graph
        .run(&[output], &values)
        .unwrap()
        .get(output)
        .unwrap()
        .to_vec();

    let target = NodeId(optimised.remap[&output.0]);
    let mut remapped = HashMap::new();
    remapped.insert(NodeId(optimised.remap[&a.0]), a_value);
    remapped.insert(NodeId(optimised.remap[&b.0]), b_value);
    let pruned = optimised
        .graph
        .run(&[target], &remapped)
        .unwrap()
        .get(target)
        .unwrap()
        .to_vec();
    assert_eq!(plain, pruned);
    assert!(!optimised.remap.contains_key(&junk.0));
}
