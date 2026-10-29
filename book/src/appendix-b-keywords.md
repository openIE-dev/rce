# Appendix B: keywords

These words are reserved and cannot be used as names (specification,
section 2.4):

```text
actor     after     and       arena     as        bound     break
chan      check     comptime  const     continue  default   device
elif      else      emit      error     event     example   extern
fail      false     fn        for       forall    from      if
impl      in        intent    iter      kind      let       linear
loop      match     module    mut       none      not       on
or        pass      pub       recv      return    select    self
send      spawn     state     step      struct    then      to
trait     true      try       type      use       var       while
with      yield
```

Some of them belong to parts of the language this book does not cover yet:
`arena` (memory regions), `comptime` (evaluation while checking), `extern`
(foreign functions), `linear` (values that must be used exactly once) and
`device` (accelerator regions). The specification describes each one.
