# Errors and refusals

RCE has no exceptions. A step either finishes or is *refused*, and a
function either returns a value or fails with a declared error.

## Refusing an event

```rce
{{#include ../listings/ch06/account.rce}}
```

- `error Insufficient(need: i64)` declares an error with a payload.
- `fail Insufficient(need: a - balance)` refuses the event with that error.
- `emit Withdrawn(amount: a)` produces an output value as part of the step.
- `event Deposit(amount: i64)` has a payload, and the step's pattern
  `Deposit(amount: a)` names it `a`.

```text
$ rce run account.rce "Deposit(amount: 50), Withdraw(amount: 20), Withdraw(amount: 40)"
{{#include ../listings/ch06/account_run.out}}
```

The third event asks for more than the balance, so it is refused and the
balance stays at 30.

## Functions that can fail

A function that can fail says so in its return type. `u8 ! NotADigit`
means "a `u8`, or the error `NotADigit`".

```rce
{{#include ../listings/ch06/parse.rce}}
```

- `fail NotADigit(at: at)` returns the error from the function.
- `try digit(c, i)` uses the value if there is one, and otherwise passes the
  error on to the caller. That is why `sum_digits` also has `! NotADigit` in
  its type.
- `sum_digits(s) catch 0` turns an error into a default value.
- `catch |e| e.at` gives the error a name so you can use its payload.

A value of type `u32 ! NotADigit` is not a `u32`. The checker refuses code
that uses it as one until the error is handled with `try` or `catch`, so an
error cannot be ignored by accident. A step whose code fails with an error
refuses the event.

## Traps

A *trap* is a failure that is not a declared error: an integer that does
not fit, a division by zero, a loop that reaches its limit. A trap inside a
step refuses the event, exactly like `fail`. A trap in a `check` or an
`example` makes the program inadmissible, as you saw in chapters 2 and 3.

## What you learned

- `error`, `fail` and refused events.
- Error unions `T ! E`, with `try`, `catch` and `catch |e|`.
- Traps refuse an event in the same way an error does.
