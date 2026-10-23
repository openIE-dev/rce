use crossterm::event::*;
use crate::InternalEvent;
pub fn code(c: &KeyCode) -> String {
    match c { KeyCode::Char(ch) => format!("c{}", *ch as u32), other => format!("{:?}", other) }
}
pub fn ev(e: &InternalEvent) -> String {
    match e {
        InternalEvent::Event(Event::Key(k)) => format!("key {} {} {} {}", code(&k.code), k.modifiers.bits(),
            match k.kind { KeyEventKind::Press => "p", KeyEventKind::Repeat => "t", KeyEventKind::Release => "r" }, k.state.bits()),
        InternalEvent::Event(Event::Mouse(m)) => format!("mouse {:?} {} {} {}", m.kind, m.column, m.row, m.modifiers.bits()),
        InternalEvent::Event(Event::FocusGained) => "focus 1".into(),
        InternalEvent::Event(Event::FocusLost) => "focus 0".into(),
        InternalEvent::Event(Event::Paste(s)) => format!("paste {}", s.chars().map(|c| (c as u32).to_string()).collect::<Vec<_>>().join(",")),
        InternalEvent::Event(Event::Resize(a, b)) => format!("resize {} {}", a, b),
        InternalEvent::CursorPosition(x, y) => format!("pos {} {}", x, y),
        InternalEvent::KeyboardEnhancementFlags(f) => format!("flags {}", f.bits()),
        InternalEvent::PrimaryDeviceAttributes => "pda".into(),
    }
}
pub trait Canon { fn canon(&self) -> String; }
impl Canon for Option<InternalEvent> { fn canon(&self) -> String { match self { None => "none".into(), Some(e) => ev(e) } } }
impl Canon for Option<char> { fn canon(&self) -> String { match self { None => "none".into(), Some(c) => format!("char {}", *c as u32) } } }
pub fn res<T: Canon>(r: &std::io::Result<T>) -> String { match r { Ok(v) => v.canon(), Err(_) => "err".into() } }
