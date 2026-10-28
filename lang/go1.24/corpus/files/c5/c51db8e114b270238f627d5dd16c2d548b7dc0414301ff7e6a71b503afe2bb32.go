// SPDX-License-Identifier: MIT OR Apache-2.0
// A binary min-heap written by hand, and container/heap on the same data.
package main

import (
	"container/heap"
	"fmt"
)

type minHeap []int

func (h *minHeap) push(x int) {
	*h = append(*h, x)
	i := len(*h) - 1
	for i > 0 {
		p := (i - 1) / 2
		if (*h)[p] <= (*h)[i] {
			break
		}
		(*h)[p], (*h)[i] = (*h)[i], (*h)[p]
		i = p
	}
}

func (h *minHeap) pop() int {
	old := *h
	top := old[0]
	n := len(old) - 1
	old[0] = old[n]
	*h = old[:n]
	i := 0
	for {
		l, r, m := 2*i+1, 2*i+2, i
		if l < n && (*h)[l] < (*h)[m] {
			m = l
		}
		if r < n && (*h)[r] < (*h)[m] {
			m = r
		}
		if m == i {
			break
		}
		(*h)[m], (*h)[i] = (*h)[i], (*h)[m]
		i = m
	}
	return top
}

type intHeap []int

func (h intHeap) Len() int           { return len(h) }
func (h intHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h intHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *intHeap) Push(x any)        { *h = append(*h, x.(int)) }
func (h *intHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

func main() {
	data := []int{9, 4, 7, 1, 8, 2, 2, 6, 3, 5, 0}
	h := &minHeap{}
	for _, d := range data {
		h.push(d)
	}
	for len(*h) > 0 {
		fmt.Print(h.pop(), " ")
	}
	fmt.Println()
	ih := &intHeap{}
	for _, d := range data {
		heap.Push(ih, d)
	}
	for ih.Len() > 0 {
		fmt.Print(heap.Pop(ih), " ")
	}
	fmt.Println()
}
