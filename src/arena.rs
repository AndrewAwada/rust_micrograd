use std::{cell::RefCell, rc::{Rc, Weak}};

pub struct Arena<V> {
    arena: Vec<Rc<RefCell<V>>>
}

impl<V> Arena<V> {
    pub fn build() -> (ArenaLifeTime<V>, ArenaRef<V>) {
        let arena = Rc::new(RefCell::new(Arena { arena: Vec::new() }));
        (ArenaLifeTime(arena.clone()), ArenaRef(Rc::downgrade(&arena)))
    }

    fn alloc(&mut self, value: V) -> Weak<RefCell<V>> {
        let shared = Rc::new(RefCell::new(value));
        self.arena.push(shared.clone());
        Rc::downgrade(&shared)
    }
}

// Arena deallocated when ArenaLifeTime struct goes out of scope. ArenaLifeTime is sole owner of Rc to Arena.
#[must_use]
pub struct ArenaLifeTime<V>(#[allow(dead_code)]Rc<RefCell<Arena<V>>>);

// User interacts with the Arena using a wrapped weak pointer.
pub struct ArenaRef<V>(Weak<RefCell<Arena<V>>>);

impl<V> ArenaRef<V> {
    // Always panic if Arena deallocated
    pub fn alloc_with_mut_borrow(&self, value: V) -> Weak<RefCell<V>> {
        let value_ptr = self.0.upgrade().expect("Arena lifetime has ended");
        let mut borrow = value_ptr.borrow_mut();
        borrow.alloc(value)
    }
}

impl<V> Clone for ArenaRef<V> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
