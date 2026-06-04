import torch                          # Core PyTorch library for tensors and autograd
import torch.nn as nn                 # Loss functions and neural network modules
import torch.optim as optim           # Optimisers (SGD, Adam, etc.)
import matplotlib.pyplot as plt       # Plotting library for the results figure
from Network import OWMNetwork, BaselineNetwork   # Our two model architectures
from data_handler import get_split_mnist          # Filtered MNIST DataLoader factory

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise fall back to CPU

def train_task(model, optimizer, criterion, dataloader, task_name, epochs=2, use_owm=False):
    """
    Trains the model on a specific task and optionally applies OWM protection.
    """
    print(f"\n--- Starting Training: {task_name} ---")
    model.train()   # Activates training-mode behaviours (dropout on, batchnorm tracking etc.)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in dataloader:                           # Iterate over every mini-batch in this task's DataLoader
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Move data to the same device as the model

            optimizer.zero_grad()          # Clear gradients from the previous batch — PyTorch accumulates by default
            outputs = model(inputs)        # Forward pass: compute logits AND cache inputs/hidden states for OWM
            loss = criterion(outputs, labels)  # Cross-entropy: measures how wrong the predictions are
            loss.backward()               # Backprop: fills .grad on every learnable parameter

            if use_owm:
                model.project_gradients()       # OWM step 1: rotate gradients into the null space of P
            optimizer.step()                    # Apply the (possibly projected) gradients to update weights
            if use_owm:
                model.update_all_projectors()   # OWM step 2: tighten P to include inputs we just learned from

            total_loss += loss.item()   # Accumulate loss; .item() detaches the scalar from the computation graph

        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss/len(dataloader):.4f}")

def evaluate_task(model, dataloader, task_name):
    """
    Tests the model's accuracy on a specific task without training.
    """
    model.eval()   # Deactivates dropout and other training-only behaviours
    correct = 0
    total = 0

    with torch.no_grad():              # Disables autograd: saves memory and speeds up inference
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)                          # Forward pass — no gradients computed
            predictions = torch.argmax(outputs, dim=1)      # Pick the class with the highest logit as the prediction
            correct += (predictions == labels).sum().item() # Count how many predictions match ground truth
            total += labels.size(0)                         # Accumulate total number of samples seen

    accuracy = 100.0 * correct / total   # Convert to percentage
    print(f"Accuracy on {task_name}: {accuracy:.2f}%")
    return accuracy

def plot_results(results):
    """Generates a comparison bar chart for OWM vs Baseline forgetting."""
    labels = ['After Task 1\n(Task 1 acc)', 'After Task 2\n(Task 1 acc)', 'After Task 2\n(Task 2 acc)']
    owm_vals  = [results['owm_t1_after_t1'],  results['owm_t1_after_t2'],  results['owm_t2_after_t2']]   # OWM accuracy at each checkpoint
    base_vals = [results['base_t1_after_t1'], results['base_t1_after_t2'], results['base_t2_after_t2']]  # Baseline accuracy at each checkpoint

    x = range(len(labels))  # Numeric positions for the three groups of bars
    width = 0.35             # Width of each individual bar; two bars side-by-side per group

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar([i - width/2 for i in x], owm_vals,  width, label='OWM',      color='#4C72B0', edgecolor='black')  # OWM bars shifted left
    bars2 = ax.bar([i + width/2 for i in x], base_vals, width, label='Baseline', color='#DD8452', edgecolor='black')  # Baseline bars shifted right

    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)                          # Extra headroom so value labels don't get clipped
    ax.set_title('OWM vs Baseline: Catastrophic Forgetting on Split-MNIST')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.axhline(y=90, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)  # Reference line at 90% for quick visual benchmarking

    for bar in bars1 + bars2:   # Annotate each bar with its exact percentage value
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('owm_results.png', dpi=150, bbox_inches='tight')  # Save to disk before showing so it's available even if display closes
    print("\nFigure saved to owm_results.png")
    plt.show()

def run_experiment():
    criterion = nn.CrossEntropyLoss()  # Standard multi-class loss; combines log-softmax + negative log-likelihood internally

    # Training loaders — shuffled, used only for weight updates
    task1_train = get_split_mnist([0, 1, 2, 3, 4], batch_size=64, train=True)
    task2_train = get_split_mnist([5, 6, 7, 8, 9], batch_size=64, train=True)
    # Test loaders — unshuffled, held-out split used only for evaluation
    task1_test  = get_split_mnist([0, 1, 2, 3, 4], batch_size=64, train=False)
    task2_test  = get_split_mnist([5, 6, 7, 8, 9], batch_size=64, train=False)

    # --- OWM Model ---
    owm_model = OWMNetwork(784, 800, 10).to(DEVICE)         # 784 inputs → 800 hidden → 10 output classes
    owm_opt   = optim.SGD(owm_model.parameters(), lr=0.01)  # Plain SGD; OWM handles forgetting, not the optimiser

    train_task(owm_model, owm_opt, criterion, task1_train, "OWM - Task 1 (Digits 0-4)", use_owm=True)
    owm_t1_after_t1 = evaluate_task(owm_model, task1_test, "OWM Task 1 (immediately after training)")   # Baseline accuracy before Task 2

    train_task(owm_model, owm_opt, criterion, task2_train, "OWM - Task 2 (Digits 5-9)", use_owm=True)
    owm_t2_after_t2 = evaluate_task(owm_model, task2_test, "OWM Task 2 (immediately after training)")   # Check Task 2 learned successfully
    owm_t1_after_t2 = evaluate_task(owm_model, task1_test, "OWM Task 1 (after learning Task 2)")        # Key metric: how much did OWM retain?

    # --- Baseline Model ---
    base_model = BaselineNetwork(784, 800, 10).to(DEVICE)         # Identical architecture, no OWM protection
    base_opt   = optim.SGD(base_model.parameters(), lr=0.01)

    train_task(base_model, base_opt, criterion, task1_train, "Baseline - Task 1 (Digits 0-4)", use_owm=False)
    base_t1_after_t1 = evaluate_task(base_model, task1_test, "Baseline Task 1 (immediately after training)")

    train_task(base_model, base_opt, criterion, task2_train, "Baseline - Task 2 (Digits 5-9)", use_owm=False)
    base_t2_after_t2 = evaluate_task(base_model, task2_test, "Baseline Task 2 (immediately after training)")
    base_t1_after_t2 = evaluate_task(base_model, task1_test, "Baseline Task 1 (after learning Task 2)")  # Should collapse — proving the problem OWM solves

    # --- Summary ---
    print("\n--- Catastrophic Forgetting Test ---")
    print(f"OWM      drop on Task 1: {owm_t1_after_t1:.1f}% → {owm_t1_after_t2:.1f}%")   # Small drop = OWM working
    print(f"Baseline drop on Task 1: {base_t1_after_t1:.1f}% → {base_t1_after_t2:.1f}%")  # Large drop = catastrophic forgetting

    plot_results({
        'owm_t1_after_t1':  owm_t1_after_t1,
        'owm_t1_after_t2':  owm_t1_after_t2,
        'owm_t2_after_t2':  owm_t2_after_t2,
        'base_t1_after_t1': base_t1_after_t1,
        'base_t1_after_t2': base_t1_after_t2,
        'base_t2_after_t2': base_t2_after_t2,
    })

if __name__ == "__main__":
    run_experiment()  # Only runs when this script is executed directly, not when imported as a module
