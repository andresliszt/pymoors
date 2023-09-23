use crate::genetic::Population;

pub fn print_minimum_objectives(population: &Population) {
    let num_objectives = population.fitness.ncols();
    let mut min_values = vec![f64::MAX; num_objectives];

    for fitness in population.fitness.outer_iter() {
        for (i, &value) in fitness.iter().enumerate() {
            if value < min_values[i] {
                min_values[i] = value;
            }
        }
    }

    println!("Minimum values of each objective:");
    for (i, &min_value) in min_values.iter().enumerate() {
        println!("Objective {}: {}", i + 1, min_value);
    }
}
