import itertools
import csv
from typing import List, Tuple
import argparse

class MathQueryGenerator:
    """
    A simplified class for generating queries based on mathematical concepts.
    """
    
    def __init__(self, concepts_file: str = None, concepts_list: List[str] = None):
        """Initialize with either a file of concepts or a list directly."""
        self.math_concepts = []
        
        if concepts_file:
            self._load_concepts_from_file(concepts_file)
        elif concepts_list:
            self.math_concepts = concepts_list
        else:
            # Default math concepts
            self.math_concepts = [
                "linear algebra", "calculus", "differential equations", 
                "topology", "number theory", "group theory", "statistics",
                "probability", "game theory", "graph theory", "combinatorics",
                "optimization", "numerical analysis", "discrete mathematics",
                "complex analysis", "real analysis", "algebra", "geometry",
                "arithmetic", "set theory", "category theory", "logic",
                "functional analysis", "dynamical systems", "information theory",
                "algebraic geometry", "cryptography", "fractals", "chaos theory",
                "Fourier analysis", "hoop strain", "vector calculus", "tensors"
            ]
            
        print(f"Loaded {len(self.math_concepts)} mathematical concepts")
    
    def _load_concepts_from_file(self, filename: str) -> None:
        """Load math concepts from a file, one concept per line."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.math_concepts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(self.math_concepts)} concepts from {filename}")
        except Exception as e:
            print(f"Error loading concepts file: {e}")
            print("Using default concept list instead")
            self.__init__()
    
    def generate_combinations(self, min_terms: int = 2, max_terms: int = 3) -> List[Tuple[str, ...]]:
        """Generate combinations of mathematical concepts."""
        all_combinations = []
        
        for r in range(min_terms, max_terms + 1):
            combinations = list(itertools.combinations(self.math_concepts, r))
            all_combinations.extend(combinations)
            print(f"Generated {len(combinations)} combinations of {r} concepts")
        
        return all_combinations
    
    def generate_queries(self, combinations: List[Tuple[str, ...]], 
                         templates: List[str] = None) -> List[str]:
        """Generate query strings from concept combinations."""
        if templates is None:
             templates = [
                "relationship between {concepts}",
                "how do {concepts} relate to each other",
                "applications of {concepts}",
                "research connecting {concepts}",
                "{concepts} unified theory",
                "proving theorems using {concepts}",
                "{concepts} practical applications",
                "innovations in {concepts}",
                "historical development of {concepts}"
            ]
        
        queries = []
        
        for combo in combinations:
            # Format concepts for insertion into templates
            if len(combo) == 1:
                concepts_str = combo[0]
            elif len(combo) == 2:
                concepts_str = f"{combo[0]} and {combo[1]}"
            else:
                concepts_str = ", ".join(combo[:-1]) + f", and {combo[-1]}"
            
            # Apply each template to the combination
            for template in templates:
                query = "Create an LLM using " + template.format(concepts=concepts_str)
                queries.append(query)
        
        print(f"Generated {len(queries)} total queries")
        return queries
    
    def save_queries(self, queries: List[str], output_file: str = "AI_constructor_queries.csv") -> None:
        """Save generated queries to a CSV file."""
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Query"])
                for query in queries:
                    writer.writerow([query])
            print(f"Queries saved to {output_file}")
        except Exception as e:
            print(f"Error saving queries: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate math concept queries")
    parser.add_argument("--concepts", help="File containing math concepts (one per line)")
    parser.add_argument("--min-terms", type=int, default=2, help="Minimum concepts per combination")
    parser.add_argument("--max-terms", type=int, default=3, help="Maximum concepts per combination")
    parser.add_argument("--output", default="math_queries.csv", help="Output file for results")
    parser.add_argument("--hoop-strain", action="store_true", help="Ensure 'hoop strain' is included in concepts")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = MathQueryGenerator(concepts_file=args.concepts)
    
    # Ensure hoop strain is in the list if requested
    if args.hoop_strain and "hoop strain" not in generator.math_concepts:
        generator.math_concepts.append("hoop strain")
        print("Added 'hoop strain' to concept list")
    
    # Generate combinations and queries
    combinations = generator.generate_combinations(args.min_terms, args.max_terms)
    queries = generator.generate_queries(combinations)
    
    # Save queries
    generator.save_queries(queries, args.output)
    
    # Print sample queries
    print("\nSample Queries:")
    for i, query in enumerate(queries[:5], 1):
        print(f"{i}. {query}")
    print(f"... and {len(queries)-5} more queries")

if __name__ == "__main__":
    main()