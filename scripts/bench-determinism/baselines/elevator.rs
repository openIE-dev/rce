//! Hand-written baseline of programs/multi/elevator.rce (two regions: the
//! car and the door; an event goes to both, and either may keep the state).
#[path = "common.rs"]
mod common;
use std::hint::black_box;

#[derive(Clone, Copy)]
enum E { Up, Down, OpenDoor, CloseDoor, Arrive }
#[derive(Debug)]
enum Kept { DoorOpen, TopFloor, GroundFloor, InMotion }
#[derive(Clone, Copy)]
struct Car { floor: u8, moving: bool }
#[derive(Clone, Copy)]
struct Door { open: bool }

fn step(car: &mut Car, door: &mut Door, e: E) -> Result<Option<u8>, Kept> {
    // every region's step reads the old state; a failure in any region
    // keeps the whole state
    let (mut c, mut d, mut out) = (*car, *door, None);
    match e {
        E::Up => {
            if door.open { return Err(Kept::DoorOpen); }
            if car.floor >= 3 { return Err(Kept::TopFloor); }
            c.floor += 1; c.moving = true; out = Some(c.floor);
        }
        E::Down => {
            if car.floor == 0 { return Err(Kept::GroundFloor); }
            if door.open { return Err(Kept::DoorOpen); }
            c.floor -= 1; c.moving = true; out = Some(c.floor);
        }
        E::OpenDoor => {
            if car.moving { return Err(Kept::InMotion); }
            d.open = true;
        }
        E::CloseDoor => d.open = false,
        E::Arrive => c.moving = false,
    }
    *car = c;
    *door = d;
    Ok(out)
}

fn pos(c: &Car, d: &Door) -> String {
    format!("[[{},{}],[{}]]", c.floor, c.moving, d.open)
}

fn main() {
    let (evs, reps) = common::input();
    let evs: Vec<E> = evs.iter().map(|e| match e.name.as_str() {
        "Up" => E::Up, "Down" => E::Down, "OpenDoor" => E::OpenDoor, "CloseDoor" => E::CloseDoor, _ => E::Arrive,
    }).collect();
    let (mut lc, mut ld, mut head) = (Car { floor: 0, moving: false }, Door { open: false }, String::new());
    for _ in 0..reps {
        let mut ch = common::Chain::new();
        let (mut car, mut door) = (Car { floor: 0, moving: false }, Door { open: false });
        for e in black_box(&evs) {
            black_box(step(&mut car, &mut door, *e).ok());
            if cfg!(bench_log) { ch.add(&format!("{{\"payload\":[],\"type\":{}}}", *e as u8), &pos(&car, &door)); }
        }
        lc = car;
        ld = door;
        head = ch.prev;
    }
    if cfg!(bench_log) { println!("head {head}"); } else { println!("state {}", pos(&lc, &ld)); }
}
