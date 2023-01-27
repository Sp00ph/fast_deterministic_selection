#![feature(
    strict_provenance,
    maybe_uninit_slice,
    get_many_mut,
    slice_swap_unchecked,
    trait_alias
)]

use std::{
    cmp::{self, Ordering},
    mem::{self, MaybeUninit},
    ptr,
    time::{Instant, Duration},
};

use rand::SeedableRng;

pub fn heapsort<T, F>(v: &mut [T], mut is_less: F)
where
    F: FnMut(&T, &T) -> bool,
{
    // This binary heap respects the invariant `parent >= child`.
    let mut sift_down = |v: &mut [T], mut node| {
        loop {
            // Children of `node`.
            let mut child = 2 * node + 1;
            if child >= v.len() {
                break;
            }

            // Choose the greater child.
            if child + 1 < v.len() && is_less(&v[child], &v[child + 1]) {
                child += 1;
            }

            // Stop if the invariant holds at `node`.
            if !is_less(&v[node], &v[child]) {
                break;
            }

            // Swap `node` with the greater child, move one step down, and continue sifting.
            v.swap(node, child);
            node = child;
        }
    };

    // Build the heap in linear time.
    for i in (0..v.len() / 2).rev() {
        sift_down(v, i);
    }

    // Pop maximal elements from the heap.
    for i in (1..v.len()).rev() {
        v.swap(0, i);
        sift_down(&mut v[..i], 0);
    }
}

/// When dropped, copies from `src` into `dest`.
struct CopyOnDrop<T> {
    src: *const T,
    dest: *mut T,
}

impl<T> Drop for CopyOnDrop<T> {
    fn drop(&mut self) {
        // SAFETY:  This is a helper class.
        //          Please refer to its usage for correctness.
        //          Namely, one must be sure that `src` and `dst` does not overlap as required by `ptr::copy_nonoverlapping`.
        unsafe {
            ptr::copy_nonoverlapping(self.src, self.dest, 1);
        }
    }
}

/// Partitions `v` into elements smaller than `pivot`, followed by elements greater than or equal
/// to `pivot`.
///
/// Returns the number of elements smaller than `pivot`.
///
/// Partitioning is performed block-by-block in order to minimize the cost of branching operations.
/// This idea is presented in the [BlockQuicksort][pdf] paper.
///
/// [pdf]: https://drops.dagstuhl.de/opus/volltexte/2016/6389/pdf/LIPIcs-ESA-2016-38.pdf
fn partition_in_blocks<T, F>(v: &mut [T], pivot: &T, is_less: &mut F) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    // Number of elements in a typical block.
    const BLOCK: usize = 128;

    // The partitioning algorithm repeats the following steps until completion:
    //
    // 1. Trace a block from the left side to identify elements greater than or equal to the pivot.
    // 2. Trace a block from the right side to identify elements smaller than the pivot.
    // 3. Exchange the identified elements between the left and right side.
    //
    // We keep the following variables for a block of elements:
    //
    // 1. `block` - Number of elements in the block.
    // 2. `start` - Start pointer into the `offsets` array.
    // 3. `end` - End pointer into the `offsets` array.
    // 4. `offsets - Indices of out-of-order elements within the block.

    // The current block on the left side (from `l` to `l.add(block_l)`).
    let mut l = v.as_mut_ptr();
    let mut block_l = BLOCK;
    let mut start_l = ptr::null_mut();
    let mut end_l = ptr::null_mut();
    let mut offsets_l = [MaybeUninit::<u8>::uninit(); BLOCK];

    // The current block on the right side (from `r.sub(block_r)` to `r`).
    // SAFETY: The documentation for .add() specifically mention that `vec.as_ptr().add(vec.len())` is always safe`
    let mut r = unsafe { l.add(v.len()) };
    let mut block_r = BLOCK;
    let mut start_r = ptr::null_mut();
    let mut end_r = ptr::null_mut();
    let mut offsets_r = [MaybeUninit::<u8>::uninit(); BLOCK];

    // FIXME: When we get VLAs, try creating one array of length `min(v.len(), 2 * BLOCK)` rather
    // than two fixed-size arrays of length `BLOCK`. VLAs might be more cache-efficient.

    // Returns the number of elements between pointers `l` (inclusive) and `r` (exclusive).
    fn width<T>(l: *mut T, r: *mut T) -> usize {
        assert!(mem::size_of::<T>() > 0);
        // FIXME: this should *likely* use `offset_from`, but more
        // investigation is needed (including running tests in miri).
        (r.addr() - l.addr()) / mem::size_of::<T>()
    }

    loop {
        // We are done with partitioning block-by-block when `l` and `r` get very close. Then we do
        // some patch-up work in order to partition the remaining elements in between.
        let is_done = width(l, r) <= 2 * BLOCK;

        if is_done {
            // Number of remaining elements (still not compared to the pivot).
            let mut rem = width(l, r);
            if start_l < end_l || start_r < end_r {
                rem -= BLOCK;
            }

            // Adjust block sizes so that the left and right block don't overlap, but get perfectly
            // aligned to cover the whole remaining gap.
            if start_l < end_l {
                block_r = rem;
            } else if start_r < end_r {
                block_l = rem;
            } else {
                // There were the same number of elements to switch on both blocks during the last
                // iteration, so there are no remaining elements on either block. Cover the remaining
                // items with roughly equally-sized blocks.
                block_l = rem / 2;
                block_r = rem - block_l;
            }
            debug_assert!(block_l <= BLOCK && block_r <= BLOCK);
            debug_assert!(width(l, r) == block_l + block_r);
        }

        if start_l == end_l {
            // Trace `block_l` elements from the left side.
            start_l = MaybeUninit::slice_as_mut_ptr(&mut offsets_l);
            end_l = start_l;
            let mut elem = l;

            for i in 0..block_l {
                // SAFETY: The unsafety operations below involve the usage of the `offset`.
                //         According to the conditions required by the function, we satisfy them because:
                //         1. `offsets_l` is stack-allocated, and thus considered separate allocated object.
                //         2. The function `is_less` returns a `bool`.
                //            Casting a `bool` will never overflow `isize`.
                //         3. We have guaranteed that `block_l` will be `<= BLOCK`.
                //            Plus, `end_l` was initially set to the begin pointer of `offsets_` which was declared on the stack.
                //            Thus, we know that even in the worst case (all invocations of `is_less` returns false) we will only be at most 1 byte pass the end.
                //        Another unsafety operation here is dereferencing `elem`.
                //        However, `elem` was initially the begin pointer to the slice which is always valid.
                unsafe {
                    // Branchless comparison.
                    *end_l = i as u8;
                    end_l = end_l.add(!is_less(&*elem, pivot) as usize);
                    elem = elem.add(1);
                }
            }
        }

        if start_r == end_r {
            // Trace `block_r` elements from the right side.
            start_r = MaybeUninit::slice_as_mut_ptr(&mut offsets_r);
            end_r = start_r;
            let mut elem = r;

            for i in 0..block_r {
                // SAFETY: The unsafety operations below involve the usage of the `offset`.
                //         According to the conditions required by the function, we satisfy them because:
                //         1. `offsets_r` is stack-allocated, and thus considered separate allocated object.
                //         2. The function `is_less` returns a `bool`.
                //            Casting a `bool` will never overflow `isize`.
                //         3. We have guaranteed that `block_r` will be `<= BLOCK`.
                //            Plus, `end_r` was initially set to the begin pointer of `offsets_` which was declared on the stack.
                //            Thus, we know that even in the worst case (all invocations of `is_less` returns true) we will only be at most 1 byte pass the end.
                //        Another unsafety operation here is dereferencing `elem`.
                //        However, `elem` was initially `1 * sizeof(T)` past the end and we decrement it by `1 * sizeof(T)` before accessing it.
                //        Plus, `block_r` was asserted to be less than `BLOCK` and `elem` will therefore at most be pointing to the beginning of the slice.
                unsafe {
                    // Branchless comparison.
                    elem = elem.sub(1);
                    *end_r = i as u8;
                    end_r = end_r.add(is_less(&*elem, pivot) as usize);
                }
            }
        }

        // Number of out-of-order elements to swap between the left and right side.
        let count = cmp::min(width(start_l, end_l), width(start_r, end_r));

        if count > 0 {
            macro_rules! left {
                () => {
                    l.add(usize::from(*start_l))
                };
            }
            macro_rules! right {
                () => {
                    r.sub(usize::from(*start_r) + 1)
                };
            }

            // Instead of swapping one pair at the time, it is more efficient to perform a cyclic
            // permutation. This is not strictly equivalent to swapping, but produces a similar
            // result using fewer memory operations.

            // SAFETY: The use of `ptr::read` is valid because there is at least one element in
            // both `offsets_l` and `offsets_r`, so `left!` is a valid pointer to read from.
            //
            // The uses of `left!` involve calls to `offset` on `l`, which points to the
            // beginning of `v`. All the offsets pointed-to by `start_l` are at most `block_l`, so
            // these `offset` calls are safe as all reads are within the block. The same argument
            // applies for the uses of `right!`.
            //
            // The calls to `start_l.offset` are valid because there are at most `count-1` of them,
            // plus the final one at the end of the unsafe block, where `count` is the minimum number
            // of collected offsets in `offsets_l` and `offsets_r`, so there is no risk of there not
            // being enough elements. The same reasoning applies to the calls to `start_r.offset`.
            //
            // The calls to `copy_nonoverlapping` are safe because `left!` and `right!` are guaranteed
            // not to overlap, and are valid because of the reasoning above.
            unsafe {
                let tmp = ptr::read(left!());
                ptr::copy_nonoverlapping(right!(), left!(), 1);

                for _ in 1..count {
                    start_l = start_l.add(1);
                    ptr::copy_nonoverlapping(left!(), right!(), 1);
                    start_r = start_r.add(1);
                    ptr::copy_nonoverlapping(right!(), left!(), 1);
                }

                ptr::copy_nonoverlapping(&tmp, right!(), 1);
                mem::forget(tmp);
                start_l = start_l.add(1);
                start_r = start_r.add(1);
            }
        }

        if start_l == end_l {
            // All out-of-order elements in the left block were moved. Move to the next block.

            // block-width-guarantee
            // SAFETY: if `!is_done` then the slice width is guaranteed to be at least `2*BLOCK` wide. There
            // are at most `BLOCK` elements in `offsets_l` because of its size, so the `offset` operation is
            // safe. Otherwise, the debug assertions in the `is_done` case guarantee that
            // `width(l, r) == block_l + block_r`, namely, that the block sizes have been adjusted to account
            // for the smaller number of remaining elements.
            l = unsafe { l.add(block_l) };
        }

        if start_r == end_r {
            // All out-of-order elements in the right block were moved. Move to the previous block.

            // SAFETY: Same argument as [block-width-guarantee]. Either this is a full block `2*BLOCK`-wide,
            // or `block_r` has been adjusted for the last handful of elements.
            r = unsafe { r.sub(block_r) };
        }

        if is_done {
            break;
        }
    }

    // All that remains now is at most one block (either the left or the right) with out-of-order
    // elements that need to be moved. Such remaining elements can be simply shifted to the end
    // within their block.

    if start_l < end_l {
        // The left block remains.
        // Move its remaining out-of-order elements to the far right.
        debug_assert_eq!(width(l, r), block_l);
        while start_l < end_l {
            // remaining-elements-safety
            // SAFETY: while the loop condition holds there are still elements in `offsets_l`, so it
            // is safe to point `end_l` to the previous element.
            //
            // The `ptr::swap` is safe if both its arguments are valid for reads and writes:
            //  - Per the debug assert above, the distance between `l` and `r` is `block_l`
            //    elements, so there can be at most `block_l` remaining offsets between `start_l`
            //    and `end_l`. This means `r` will be moved at most `block_l` steps back, which
            //    makes the `r.offset` calls valid (at that point `l == r`).
            //  - `offsets_l` contains valid offsets into `v` collected during the partitioning of
            //    the last block, so the `l.offset` calls are valid.
            unsafe {
                end_l = end_l.sub(1);
                ptr::swap(l.add(usize::from(*end_l)), r.sub(1));
                r = r.sub(1);
            }
        }
        width(v.as_mut_ptr(), r)
    } else if start_r < end_r {
        // The right block remains.
        // Move its remaining out-of-order elements to the far left.
        debug_assert_eq!(width(l, r), block_r);
        while start_r < end_r {
            // SAFETY: See the reasoning in [remaining-elements-safety].
            unsafe {
                end_r = end_r.sub(1);
                ptr::swap(l, r.sub(usize::from(*end_r) + 1));
                l = l.add(1);
            }
        }
        width(v.as_mut_ptr(), l)
    } else {
        // Nothing else to do, we're done.
        width(v.as_mut_ptr(), l)
    }
}

/// Partitions `v` into elements smaller than `v[pivot]`, followed by elements greater than or
/// equal to `v[pivot]`.
///
/// Returns a tuple of:
///
/// 1. Number of elements smaller than `v[pivot]`.
/// 2. True if `v` was already partitioned.
fn partition<T, F>(v: &mut [T], pivot: usize, is_less: &mut F) -> (usize, bool)
where
    F: FnMut(&T, &T) -> bool,
{
    let (mid, was_partitioned) = {
        // Place the pivot at the beginning of slice.
        v.swap(0, pivot);
        let (pivot, v) = v.split_at_mut(1);
        let pivot = &mut pivot[0];

        // Read the pivot into a stack-allocated variable for efficiency. If a following comparison
        // operation panics, the pivot will be automatically written back into the slice.

        // SAFETY: `pivot` is a reference to the first element of `v`, so `ptr::read` is safe.
        let tmp = mem::ManuallyDrop::new(unsafe { ptr::read(pivot) });
        let _pivot_guard = CopyOnDrop {
            src: &*tmp,
            dest: pivot,
        };
        let pivot = &*tmp;

        // Find the first pair of out-of-order elements.
        let mut l = 0;
        let mut r = v.len();

        // SAFETY: The unsafety below involves indexing an array.
        // For the first one: We already do the bounds checking here with `l < r`.
        // For the second one: We initially have `l == 0` and `r == v.len()` and we checked that `l < r` at every indexing operation.
        //                     From here we know that `r` must be at least `r == l` which was shown to be valid from the first one.
        unsafe {
            // Find the first element greater than or equal to the pivot.
            while l < r && is_less(v.get_unchecked(l), pivot) {
                l += 1;
            }

            // Find the last element smaller that the pivot.
            while l < r && !is_less(v.get_unchecked(r - 1), pivot) {
                r -= 1;
            }
        }

        (
            l + partition_in_blocks(&mut v[l..r], pivot, is_less),
            l >= r,
        )

        // `_pivot_guard` goes out of scope and writes the pivot (which is a stack-allocated
        // variable) back into the slice where it originally was. This step is critical in ensuring
        // safety!
    };

    // Place the pivot between the two partitions.
    v.swap(0, mid);

    (mid, was_partitioned)
}

/// Shifts the last element to the left until it encounters a smaller or equal element.
fn shift_tail<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();
    // SAFETY: The unsafe operations below involves indexing without a bound check (by offsetting a
    // pointer) and copying memory (`ptr::copy_nonoverlapping`).
    //
    // a. Indexing:
    //  1. We checked the size of the array to >= 2.
    //  2. All the indexing that we will do is always between `0 <= index < len-1` at most.
    //
    // b. Memory copying
    //  1. We are obtaining pointers to references which are guaranteed to be valid.
    //  2. They cannot overlap because we obtain pointers to difference indices of the slice.
    //     Namely, `i` and `i+1`.
    //  3. If the slice is properly aligned, the elements are properly aligned.
    //     It is the caller's responsibility to make sure the slice is properly aligned.
    //
    // See comments below for further detail.
    unsafe {
        // If the last two elements are out-of-order...
        if len >= 2 && is_less(v.get_unchecked(len - 1), v.get_unchecked(len - 2)) {
            // Read the last element into a stack-allocated variable. If a following comparison
            // operation panics, `hole` will get dropped and automatically write the element back
            // into the slice.
            let tmp = mem::ManuallyDrop::new(ptr::read(v.get_unchecked(len - 1)));
            let v = v.as_mut_ptr();
            let mut hole = CopyOnDrop {
                src: &*tmp,
                dest: v.add(len - 2),
            };
            ptr::copy_nonoverlapping(v.add(len - 2), v.add(len - 1), 1);

            for i in (0..len - 2).rev() {
                if !is_less(&*tmp, &*v.add(i)) {
                    break;
                }

                // Move `i`-th element one place to the right, thus shifting the hole to the left.
                ptr::copy_nonoverlapping(v.add(i), v.add(i + 1), 1);
                hole.dest = v.add(i);
            }
            // `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
        }
    }
}

/// Sorts a slice using insertion sort, which is *O*(*n*^2) worst-case.
fn insertion_sort<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    for i in 1..v.len() {
        shift_tail(&mut v[..i + 1], is_less);
    }
}

pub trait Pred<T> = FnMut(&T, &T) -> bool;

fn from_is_less<T>(
    is_less: &mut impl Pred<T>,
) -> impl FnMut(&(usize, &T), &(usize, &T)) -> Ordering + '_ {
    |&(_, x), &(_, y)| {
        if is_less(x, y) {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

pub fn fds<T, F: Pred<T>>(mut v: &mut [T], is_less: &mut F, mut k: usize) {
    if k >= v.len() {
        panic!("Out of bounds");
    }
    if mem::size_of::<T>() == 0 {
        return;
    }
    // We now know that `k < v.len() <= isize::MAX`
    loop {
        if v.len() <= 16 {
            insertion_sort(v, is_less);
            return;
        }

        if k == v.len() - 1 {
            // Find max element and place it in the last position of the array. We're free to use
            // `unwrap()` here because we know v must not be empty.
            let (max_index, _) = v.iter().enumerate().max_by(from_is_less(is_less)).unwrap();
            v.swap(max_index, k);
            return;
        } else if k == 0 {
            // Find min element and place it in the first position of the array. We're free to use
            // `unwrap()` here because we know v must not be empty.
            let (min_index, _) = v.iter().enumerate().min_by(from_is_less(is_less)).unwrap();
            v.swap(min_index, k);
            return;
        }

        let six_k = k.saturating_mul(6);
        let p = if six_k < v.len() {
            median_of_minima(v, is_less, k)
        } else if six_k > v.len().saturating_mul(5) {
            median_of_maxima(v, is_less, k)
        } else {
            median_of_ninthers(v, is_less)
        };

        if p == k {
            return;
        } else if p > k {
            v = &mut v[..p];
        } else {
            v = &mut v[p + 1..];
            k -= p + 1;
        }
    }
}

fn median_of_minima<T, F: Pred<T>>(v: &mut [T], is_less: &mut F, k: usize) -> usize {
    debug_assert!(k > 0);

    let subset = k * 2;
    let compute_min_over = v.len() / subset - 1;

    debug_assert!(compute_min_over > 0);

    let mut j = subset;
    for i in 0..subset {
        let limit = j + compute_min_over;

        let min_idx = v[j..limit]
            .iter()
            .enumerate()
            .min_by(from_is_less(is_less))
            .map_or(j, |(i, _)| i + j);

        if is_less(&v[min_idx], &v[i]) {
            v.swap(min_idx, i)
        }

        j = limit;
    }

    fds(&mut v[..subset], is_less, k);
    partition(v, k, is_less).0
}

fn median_of_maxima<T, F: Pred<T>>(v: &mut [T], is_less: &mut F, k: usize) -> usize {
    let subset = (v.len() - k) * 2;
    let subset_start = v.len() - subset;
    let compute_max_over = subset_start / subset;
    debug_assert!(compute_max_over > 0);

    let mut j = subset_start - subset * compute_max_over;
    for i in subset_start..v.len() {
        let limit = j + compute_max_over;

        let max_idx = v[j..limit]
            .iter()
            .enumerate()
            .max_by(from_is_less(is_less))
            .map_or(j, |(i, _)| i + j);

        if is_less(&v[i], &v[max_idx]) {
            v.swap(i, max_idx);
        }

        j = limit;
    }

    let len = v.len();
    fds(&mut v[subset_start..], is_less, len - k);
    partition(v, k, is_less).0
}

fn median_of_ninthers<T, F: Pred<T>>(v: &mut [T], is_less: &mut F) -> usize {
    let frac = if v.len() <= 1024 {
        v.len() / 12
    } else if v.len() <= 128 * 1024 {
        v.len() / 64
    } else {
        v.len() / 1024
    };

    let pivot = frac / 2;
    let lo = v.len() / 2 - pivot;
    let hi = frac + lo;
    let gap = (v.len() - 9 * frac) / 4;
    let mut a = lo - 4 * frac - gap;
    let mut b = hi + gap;
    for i in lo..hi {
        // don't question it.
        ninther(
            v,
            is_less,
            a,
            i - frac,
            b,
            a + 1,
            i,
            b + 1,
            a + 2,
            i + frac,
            b + 2,
        );
        a += 3;
        b += 3;
    }

    fds(&mut v[lo..lo + frac], is_less, pivot);
    // dbg!(v.len(), lo + pivot);
    partition(v, lo + pivot, is_less).0
}

fn ninther<T, F: Pred<T>>(
    v: &mut [T],
    is_less: &mut F,
    a: usize,
    mut b: usize,
    c: usize,
    mut d: usize,
    e: usize,
    mut f: usize,
    g: usize,
    mut h: usize,
    i: usize,
) {
    b = median_idx(v, is_less, a, b, c);
    h = median_idx(v, is_less, g, h, i);
    if is_less(&v[h], &v[b]) {
        mem::swap(&mut b, &mut h);
    }
    if is_less(&v[f], &v[d]) {
        mem::swap(&mut d, &mut f);
    }
    if is_less(&v[e], &v[d]) {

    } else if is_less(&v[f], &v[e]) {
        d = f;
    } else {
        if is_less(&v[e], &v[b]) {
            v.swap(e, b);
        } else if is_less(&v[h], &v[e]) {
            v.swap(e, h);
        }
        return;
    }
    if is_less(&v[d], &v[b]) {
        d = b;
    } else if is_less(&v[h], &v[d]) {
        d = h;
    }

    v.swap(d, e);
}

fn median_idx<T, F: Pred<T>>(
    v: &[T],
    is_less: &mut F,
    mut a: usize,
    b: usize,
    mut c: usize,
) -> usize {
    if is_less(&v[c], &v[a]) {
        mem::swap(&mut a, &mut c);
    }
    if is_less(&v[c], &v[b]) {
        return c;
    }
    if is_less(&v[b], &v[a]) {
        return a;
    }
    b
}

fn main() {
    use rand::{prelude::SliceRandom, rngs::SmallRng, thread_rng};
    let mut rng = SmallRng::from_rng(thread_rng()).unwrap();
    let len = 1 << 16;
    let idx = len / 16;
    let runs = 100;
    let mut v = Vec::from_iter(0..len);
    v.shuffle(&mut rng);
    let v = v;
    let mut buf = vec![0; v.len()];

    let mut d = Duration::ZERO;
    for _ in 0..runs {
        buf.clone_from(&v);
        let t = Instant::now();
        heapsort(&mut buf, usize::lt);
        d += t.elapsed();
    }
    println!("heapsort: {:.2?}", d / runs);
    assert_eq!(buf[idx], idx);
    assert!(buf[..idx].iter().all(|&i| i < idx) && buf[idx+1..].iter().all(|&i| i > idx));
    buf.sort_unstable();
    assert!(buf.iter().copied().eq(0..len));
    
    
    let mut d = Duration::ZERO;
    for _ in 0..runs {
        buf.clone_from(&v);
        let t = Instant::now();
        fds(&mut buf, &mut usize::lt, idx);
        d += t.elapsed();
    }
    println!("fast deterministic selection: {:.2?}", d / runs);
    // assert the select postconditions
    assert_eq!(buf[idx], idx);
    assert!(buf[..idx].iter().all(|&i| i < idx) && buf[idx+1..].iter().all(|&i| i > idx));
    // make sure that the select was a permutation, so all original items are still in there
    buf.sort_unstable();
    assert!(buf.iter().copied().eq(0..len));
}
