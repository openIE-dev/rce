// SPDX-License-Identifier: MIT OR Apache-2.0
// Goroutines and channels whose output does not depend on the schedule:
// a pipeline over unbuffered channels, and results gathered by index.
package main

import (
	"fmt"
	"sync"
)

func gen(n int) <-chan int {
	out := make(chan int)
	go func() {
		for i := 1; i <= n; i++ {
			out <- i
		}
		close(out)
	}()
	return out
}

func square(in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		for v := range in {
			out <- v * v
		}
		close(out)
	}()
	return out
}

func main() {
	for v := range square(gen(6)) {
		fmt.Print(v, " ")
	}
	fmt.Println()
	results := make([]int, 8)
	var wg sync.WaitGroup
	for i := range results {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := 0
			for k := 0; k <= i*1000; k++ {
				s += k
			}
			results[i] = s
		}()
	}
	wg.Wait()
	fmt.Println(results)
	var mu sync.Mutex
	total := 0
	var wg2 sync.WaitGroup
	for i := 1; i <= 100; i++ {
		wg2.Add(1)
		go func(n int) {
			defer wg2.Done()
			mu.Lock()
			total += n
			mu.Unlock()
		}(i)
	}
	wg2.Wait()
	fmt.Println(total)
	buf := make(chan string, 3)
	buf <- "a"
	buf <- "b"
	fmt.Println(len(buf), cap(buf), <-buf, <-buf)
	select {
	case v := <-buf:
		fmt.Println("got", v)
	default:
		fmt.Println("empty")
	}
	close(buf)
	v, ok := <-buf
	fmt.Printf("%q %v\n", v, ok)
}
