# 🎲 Nash Equilibrium for Liar's Dice with Llamas (Jokers)

This project calculates and plays a **Nash equilibrium strategy** for a modified version of **Liar's Dice**, where `1`s are replaced by **llamas ("L")**, which act as wildcards (jokers). The strategy is trained for **1-vs-1** play with **one die each**, but the framework supports larger configurations given sufficient compute time.

You can:
- 🔹 Play against a computer using Nash equilibrium
- 🔹 Use real dice to play manually
- 🔹 See which of your moves were optimal
- 🔹 Optionally include a **Call ("C")** action (code present but not trained)

> **Just run [`play.py`](./play.py) to play.**

---

## 🕹️ Gameplay Rules

- Each player secretly rolls their die.
- Players alternate making **increasing bets**:
  - Bet must increase in quantity or face value.
  - `"L"` (Llama) is a **wild** face and counts as any number.
  - **Cannot start** with a llama bet.
- At any point, you may:
  - **Doubt** the opponent’s last bet by choosing `"D"` — if their claim is false, you win.
  - Use **Call `"C"`** (optional, logic present but not yet trained) to challenge an exact match.
- The loser of a challenge loses the round.

---

## 🚀 Quick Start

1. Clone the repo:
   ```bash
   git clone https://github.com/RyanCotsakis/liars_dice.git
   cd liars-dice
   ```

2. Install dependencies:
   ```bash
   pip install torch numpy pandas
   ```

3. Play against the equilibrium strategy:
   ```bash
   python play.py
   ```

---

## 🧠 How It Works

- The Nash equilibrium strategy is stored in `grandfather.pkl`, a serialized object using Python’s `pickle` module.
- This file contains the **head node of a linked structure** representing all reachable game states.
- Each node contains:
  - A probability matrix (`logits`) mapping a player's **private roll** to their **optimal move distribution**.
  - Children nodes representing legal subsequent moves.
- During training, strategies for each player are updated using gradient descent over this tree.

---

## 🛠️ Training the Equilibrium (Optional)

You can retrain or extend the equilibrium strategy by changing the dice configuration:

1. Edit this in `play.py`:
   ```python
   N_DICE = (1, 1)  # Change to (2,2), (3,3), etc.
   ```

2. Then run the training loop:
   ```python
   calculate_equilibrium()
   ```

This will:
- Build the full game tree
- Use self-play and policy optimization to approximate the equilibrium
- Save the resulting policy to `grandfather.pkl`

---

## 📦 Key Components

- `play.py`: Main entry point to play or train
- `Die`: Rolls dice with llama logic
- `Player`: Human or AI with strategy lookup
- `Node`: Represents game tree node with outcome logic
- `grandfather.pkl`: Serialized game state tree with optimal policies
