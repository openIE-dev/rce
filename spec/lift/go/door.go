package door

import "errors"

type Door struct {
	open   bool
	locked bool
}

var ErrDoorOpen = errors.New("DoorOpen")
var ErrLocked = errors.New("Locked")

func (d *Door) Open() error {
	if d.locked {
		return ErrLocked
	}
	d.open = true
	return nil
}

func (d *Door) Close() error {
	d.open = false
	return nil
}

func (d *Door) Lock() error {
	if d.open {
		return ErrDoorOpen
	}
	d.locked = true
	return nil
}

func (d *Door) Unlock() error {
	d.locked = false
	return nil
}
