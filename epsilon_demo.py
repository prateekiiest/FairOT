import numpy as np

# Simple demonstration of epsilon in extended matrix for partial OT
def demonstrate_epsilon_extended_matrix():
    """
    Demonstrates how epsilon appears in the extended matrix approach for partial OT.
    """
    print("=" * 60)
    print("DEMONSTRATION: EPSILON IN EXTENDED MATRIX")
    print("=" * 60)
    
    # Create a simple 2x3 similarity matrix
    S_sub = np.array([
        [0.8, 0.6, 0.4],
        [0.7, 0.9, 0.5]
    ])
    
    print(f"Original similarity matrix S_sub:")
    print(S_sub)
    print()
    
    # Parameters
    epsilon = 1.0
    beta = 2.0  # Large constant
    
    print(f"Parameters: epsilon = {epsilon}, beta = {beta}")
    print()
    
    # Step 1: Original cost matrix C = -S_sub
    C_original = -S_sub
    print(f"Original cost matrix C = -S_sub:")
    print(C_original)
    print()
    
    # Step 2: Extended similarity matrix according to paper
    # S_tilde_{ij} = beta - C_{ij} = beta - (-S_{ij}) = beta + S_{ij}
    S_extended_original = beta + S_sub
    print(f"Extended similarity (original part) = beta + S_sub:")
    print(S_extended_original)
    print()
    
    # Step 3: Dummy column similarity
    # We want dummy cost = epsilon, so dummy similarity = beta - epsilon
    dummy_similarity = beta - epsilon
    S_dummy_column = np.full((S_sub.shape[0], 1), dummy_similarity)
    print(f"Dummy column similarity = beta - epsilon = {beta} - {epsilon} = {dummy_similarity}:")
    print(S_dummy_column.flatten())
    print()
    
    # Step 4: Complete extended similarity matrix
    S_extended = np.hstack([S_extended_original, S_dummy_column])
    print(f"Complete extended similarity matrix S_tilde:")
    print(S_extended)
    print()
    
    # Step 5: Convert back to cost matrix for OT solver
    C_extended = -S_extended
    print(f"Extended cost matrix C_tilde = -S_tilde:")
    print(C_extended)
    print()
    
    # Verify: dummy column should have cost = epsilon
    dummy_costs = C_extended[:, -1]
    print(f"Dummy column costs: {dummy_costs} (should all be {epsilon})")
    print(f"✓ Dummy costs are epsilon: {np.allclose(dummy_costs, epsilon)}")
    print()
    
    return S_extended, C_extended

if __name__ == "__main__":
    demonstrate_epsilon_extended_matrix()
