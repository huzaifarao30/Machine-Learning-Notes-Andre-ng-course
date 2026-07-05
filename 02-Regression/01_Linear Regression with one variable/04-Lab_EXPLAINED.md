# Linear Regression Lab — Line-by-Line Explanation

This file explains EVERY line in `linear_regression_lab.py`, in order.
Read it side-by-side with the code file.

---

## 1. Imports

```python
import numpy as np
```
Loads NumPy, and lets you call its functions using the short name `np`.
NumPy is used for arrays and number-crunching.

```python
import matplotlib.pyplot as plt
```
Loads Matplotlib's plotting module, short name `plt`. Used to draw graphs.

```python
plt.style.use('./deeplearning.mplstyle')
```
Applies a pre-made visual style (colors, fonts) from the course's style file.
Purely cosmetic — has zero effect on the ML logic. If this file is missing on
your machine, you can comment this line out and everything still works.

---

## 2. Creating the data

```python
x_train = np.array([1.0, 2.0])
```
Creates a NumPy array holding your input feature — house sizes, in units of
1000 sqft. So `1.0` = 1000 sqft, `2.0` = 2000 sqft.

```python
y_train = np.array([300.0, 500.0])
```
Creates a NumPy array holding your target/output — house prices, in units of
$1000. So `300.0` = $300,000, `500.0` = $500,000.

```python
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
```
`f"...{x_train}..."` is an f-string — anything inside `{}` gets evaluated and
inserted into the string. These two lines just print the arrays so you can
visually confirm what's in them.

---

## 3. Counting training examples (`m`)

```python
print(f"x_train.shape: {x_train.shape}")
```
`.shape` is a NumPy array attribute that returns a tuple describing the
array's dimensions. For a 1D array of 2 numbers, this prints `(2,)`.

```python
m = x_train.shape[0]
```
`shape[0]` grabs the first number in that tuple — the length of the array.
Here, `m` becomes `2` (2 training examples).

```python
print(f"Number of training examples is: {m}")
```
Prints the value of `m`.

```python
m = len(x_train)
```
A second, simpler way to get the same result — Python's built-in `len()`
function also returns the number of elements in the array. `m` is still `2`.

```python
print(f"Number of training examples is: {m}")
```
Prints it again, just to show both methods give the same answer.

---

## 4. Accessing a single training example

```python
i = 0
```
Sets an index variable. Python counts from 0, so index `0` = the 1st example.

```python
x_i = x_train[i]
y_i = y_train[i]
```
Square brackets `[i]` pull out the value at position `i` from the array.
Since `i = 0`: `x_i = 1.0` and `y_i = 300.0` — the first house's size and price.

```python
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
```
Prints it in the notation form (x⁽⁰⁾, y⁽⁰⁾) = (1.0, 300.0), matching the math
notation from your notes.

*(To see the 2nd example, you'd change `i = 0` to `i = 1`.)*

---

## 5. Plotting the raw data

```python
plt.scatter(x_train, y_train, marker='x', c='r')
```
`plt.scatter(x, y, ...)` plots individual points — x-values from `x_train`,
y-values from `y_train`. `marker='x'` draws each point as an X shape,
`c='r'` colors them red.

```python
plt.title("Housing Prices")
```
Sets the graph's title text.

```python
plt.ylabel('Price (in 1000s of dollars)')
```
Labels the vertical (y) axis.

```python
plt.xlabel('Size (1000 sqft)')
```
Labels the horizontal (x) axis.

```python
plt.show()
```
Actually renders/displays the graph on screen. Without this line, nothing
would visually appear — you'd just be building the plot in memory.

---

## 6. Setting model parameters (first guess)

```python
w = 100
b = 100
```
Sets the model's two parameters: `w` (weight/slope) and `b` (bias/intercept).
These are just guesses to start — you'll see they're wrong ones.

```python
print(f"w: {w}")
print(f"b: {b}")
```
Prints the current values of `w` and `b`.

---

## 7. The model function

```python
def compute_model_output(x, w, b):
```
Defines a function named `compute_model_output` that takes 3 inputs:
`x` (an array of sizes), `w` and `b` (the two model parameters).

```python
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w,b (scalar)    : model parameters
    Returns
      f_wb (ndarray (m,)): model prediction
    """
```
This is a docstring — a comment block describing what the function does,
its inputs, and its output. Doesn't run any code; it's documentation only.

```python
    m = x.shape[0]
```
Finds how many data points are in `x` (reuses the `.shape[0]` trick from
earlier), so the function knows how many predictions to make.

```python
    f_wb = np.zeros(m)
```
Creates an array of `m` zeros — an empty container to be filled in with
predictions. E.g., for `m=2` this creates `[0.0, 0.0]`.

```python
    for i in range(m):
```
Starts a loop that runs once for each index `i` from `0` up to `m-1`
(so for m=2: i=0, then i=1).

```python
        f_wb[i] = w * x[i] + b
```
This is the actual linear regression formula: **f(x) = wx + b**.
For each data point `x[i]`, it multiplies by `w`, adds `b`, and stores the
result at position `i` in the `f_wb` array.

```python
    return f_wb
```
Sends the completed array of predictions back to whoever called the function.

---

## 8. Using the model function (with the wrong w, b)

```python
tmp_f_wb = compute_model_output(x_train, w, b,)
```
Calls the function with your data and current `w=100, b=100`. Stores the
resulting predictions in `tmp_f_wb`. (The trailing comma before `)` is
harmless — Python allows it.)

```python
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
```
`plt.plot(x, y, ...)` draws a connected LINE (not separate dots) through the
given points. Here it draws your model's prediction line, in blue (`c='b'`),
and tags it "Our Prediction" for the legend.

```python
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
```
Re-plots the actual red data points on the same graph, labeled "Actual Values".

```python
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
```
Same labeling as before.

```python
plt.legend()
```
Displays a small legend box on the graph showing which color/line means what
(using the `label=` text you set above).

```python
plt.show()
```
Renders the graph. At this point you'd visually see the blue line does NOT
pass through the red X's — proof that `w=100, b=100` is a bad fit.

---

## 9. The Challenge — finding the right w and b

The lab challenge asks you to find values of `w` and `b` that make the line
actually pass through both data points: (1, 300) and (2, 500).

**How to solve it manually:**
- Slope `w` = (change in y) / (change in x) = (500 − 300) / (2 − 1) = **200**
- Plug into f(x) = wx + b using point (1, 300): 200×(1) + b = 300 → **b = 100**

```python
w = 200
b = 100
```
Updates the parameters to the correct values.

```python
tmp_f_wb = compute_model_output(x_train, w, b)
```
Recomputes predictions using the corrected `w` and `b`.

```python
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
```
Same plotting steps as before. This time, the blue line passes exactly
through both red X's — a perfect fit for this tiny dataset.

---

## 10. Making a prediction with the fitted model

```python
w = 200
b = 100
```
Re-confirms the correct parameters (in case you're running this section on
its own).

```python
x_i = 1.2
```
A NEW input — a house of 1200 sqft. Since units are in 1000s of sqft, this
is written as `1.2`.

```python
cost_1200sqft = w * x_i + b
```
Applies the model formula directly: 200 × 1.2 + 100 = 340.

```python
print(f"${cost_1200sqft:.0f} thousand dollars")
```
Prints the result. The `:.0f` inside the f-string formats the number as a
whole number (no decimals). Output: `$340 thousand dollars`.

This is the entire point of the lab: once you have the right `w` and `b`,
you can predict the price of houses that weren't even in your original data.

---

## Summary of the flow

1. Import tools (NumPy for math, Matplotlib for graphs)
2. Store your data as NumPy arrays
3. Learn to count and access data points (`m`, indexing)
4. Visualize the raw data
5. Define the model formula f(x) = wx + b as a Python function
6. Try bad parameters → see a bad fit
7. Manually solve for the correct w, b → see a perfect fit
8. Use the fitted model to predict something new

Later in the course, instead of solving for `w` and `b` by hand (like you
just did), you'll learn a **cost function** (to measure how wrong a line is)
and **gradient descent** (an algorithm that automatically adjusts w and b to
minimize that wrongness). This lab is designed to make you feel why that's
useful — you just did manually what gradient descent will soon do for you,
automatically, on datasets far too big to solve by hand.
