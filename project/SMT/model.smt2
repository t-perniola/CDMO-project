; Declare sorts
; No additional sorts needed beyond Int and Bool

; Declare functions and arrays
(declare-fun path (Int Int) Int)
(declare-fun path_length (Int) Int)
(declare-fun total_distance (Int) Int)
(declare-fun b_path (Int Int) Bool)

; Define constants and variables
(declare-const m Int)
(declare-const n Int)
(declare-const l (Array Int Int))
(declare-const s (Array Int Int))
(declare-const D (Array Int (Array Int Int)))

; Read input data from file or assert directly
(assert (= m 2)) ; Replace with the actual value of m from inst01.dat
(assert (= n 6)) ; Replace with the actual value of n from inst01.dat

; Load capacities l[c-1]
(assert (= (select l 0) 15))
(assert (= (select l 1) 10))

; Items' sizes s[i-1]
(assert (= (select s 0) 3))
(assert (= (select s 1) 2))
(assert (= (select s 2) 6))
(assert (= (select s 3) 5))
(assert (= (select s 4) 4))
(assert (= (select s 5) 4))

; Distance matrix D[i][j]
(assert (= (select (select D 0) 0) 0))
(assert (= (select (select D 0) 1) 3))
(assert (= (select (select D 0) 2) 4))
(assert (= (select (select D 0) 3) 5))
(assert (= (select (select D 0) 4) 6))
(assert (= (select (select D 0) 5) 6))
(assert (= (select (select D 1) 0) 3))
(assert (= (select (select D 1) 1) 0))
(assert (= (select (select D 1) 2) 1))
(assert (= (select (select D 1) 3) 4))
(assert (= (select (select D 1) 4) 5))
(assert (= (select (select D 1) 5) 7))
(assert (= (select (select D 2) 0) 4))
(assert (= (select (select D 2) 1) 1))
(assert (= (select (select D 2) 2) 0))
(assert (= (select (select D 2) 3) 5))
(assert (= (select (select D 2) 4) 6))
(assert (= (select (select D 2) 5) 4))
(assert (= (select (select D 3) 0) 5))
(assert (= (select (select D 3) 1) 4))
(assert (= (select (select D 3) 2) 5))
(assert (= (select (select D 3) 3) 0))
(assert (= (select (select D 3) 4) 3))
(assert (= (select (select D 3) 5) 3))
(assert (= (select (select D 4) 0) 6))
(assert (= (select (select D 4) 1) 7))
(assert (= (select (select D 4) 2) 8))
(assert (= (select (select D 4) 3) 3))
(assert (= (select (select D 4) 4) 0))
(assert (= (select (select D 4) 5) 2))
(assert (= (select (select D 5) 0) 6))
(assert (= (select (select D 5) 1) 7))
(assert (= (select (select D 5) 2) 8))
(assert (= (select (select D 5) 3) 3))
(assert (= (select (select D 5) 4) 2))
(assert (= (select (select D 5) 5) 0))

; Define maximum number of items per courier
(declare-const MAX_ITEMS Int)
(assert (= MAX_ITEMS (+ (div n m) 3)))

; Compute MAX_Couriers and MAX_Items
(declare-const MAX_Couriers Int)
(declare-const MAX_Items Int)
(assert (= MAX_Couriers (+ m 1)))
(assert (= MAX_Items (+ n 1)))

; Define the number of couriers and items
(declare-const Couriers (Array Int Bool))
(declare-const Items (Array Int Bool))

; Initialize Couriers and Items arrays with false
(assert (forall ((i Int)) (= (select Couriers i) false)))
(assert (forall ((i Int)) (= (select Items i) false)))

; Constraints for Couriers and Items ranges
(assert (forall ((i Int)) (=> (and (>= i 1) (<= i MAX_Couriers)) (select Couriers i))))
(assert (forall ((i Int)) (=> (and (>= i 1) (<= i MAX_Items)) (select Items i))))

; Paths should range between 0 and n+1
(assert (forall ((c Int) (j Int)) (=> (and (>= c 1) (<= c m) (>= j 1) (<= j MAX_ITEMS))
                                      (and (>= (path c j) 0) (<= (path c j) (+ n 1))))))

; Boundaries of path_length's values
(assert (forall ((c Int)) (=> (and (>= c 1) (<= c m))
                              (and (>= (path_length c) 3) (<= (path_length c) MAX_ITEMS)))))

; Initial and final node constraints
(assert (forall ((c Int)) (=> (and (>= c 1) (<= c m))
                              (and (= (path c 1) (+ n 1))
                                   (= (path c (path_length c)) (+ n 1))))))

; Set unvisited items to zero
(assert (forall ((c Int) (i Int)) (=> (and (>= c 1) (<= c m) (> i (path_length c)))
                                      (= (path c i) 0))))

; Constraints for load capacity
(assert (forall ((c Int)) (=> (and (>= c 1) (<= c m))
  (<= (+ (ite (b_path c 1) (select s 0) 0)
         (ite (b_path c 2) (select s 1) 0)
         (ite (b_path c 3) (select s 2) 0)
         (ite (b_path c 4) (select s 3) 0)
         (ite (b_path c 5) (select s 4) 0)
         (ite (b_path c 6) (select s 5) 0))
     (select l (- c 1))))))

; Constraints for item assignment
(assert (forall ((j Int)) (=> (and (>= j 1) (<= j n))
  (= (+ (ite (b_path 1 j) 1 0)
         (ite (b_path 2 j) 1 0)
         (ite (b_path 3 j) 1 0)
         (ite (b_path 4 j) 1 0)
         (ite (b_path 5 j) 1 0)
         (ite (b_path 6 j) 1 0))
     1))))

; Constraints for maximum items per courier
(assert (forall ((c Int)) (=> (and (>= c 1) (<= c m))
  (<= (+ (ite (b_path c 1) 1 0)
         (ite (b_path c 2) 1 0)
         (ite (b_path c 3) 1 0)
         (ite (b_path c 4) 1 0)
         (ite (b_path c 5) 1 0)
         (ite (b_path c 6) 1 0))
     MAX_ITEMS))))

; Distance computation
(assert (forall ((c Int)) (=> (and (>= c 1) (<= c m))
  (= (total_distance c)
     (+ (compute-forward-distance c)
        (compute-backward-distance c))))))

(define-fun compute-forward-distance ((c Int)) Int
  (let ((result 0))
    (forall ((j Int))
      (when (and (<= j (- (path_length c) 1))
                 (not (= (path c j) 0))
                 (not (= (path c (+ j 1)) 0)))
        (setq result (+ result (select (select D (- (path c j) 1)) (- (path c (+ j 1)) 1))))))
    result))

(define-fun compute-backward-distance ((c Int)) Int
  (let ((result 0))
    (forall ((j Int))
      (when (and (>= j 1)
                 (= (path c j) 0)
                 (not (= (path c (- j 1)) 0))
                 (not (= (path c (+ j 1)) 0)))
        (setq result (+ result (select (select D (- (path c (- j 1)) 1)) (- (path c (+ j 1)) 1))))))
    result))

; Symmetry breaking: two couriers with the same load size
(assert (forall ((c1 Int) (c2 Int))
  (=> (and (>= c1 1) (<= c1 m) (>= c2 1) (<= c2 m) (< c1 c2))
      (ite (= (select l (- c1 1)) (select l (- c2 1)))
           (lexleq (map (lambda ((j Int)) (b_path c1 j)) (range 1 (+ n 1)))
                   (map (lambda ((j Int)) (b_path c2 j)) (range 1 (+ n 1))))
           true))))

; Optimization objective: minimize the maximum distance traveled by any courier
(minimize (max (map (lambda ((c Int)) (total_distance c)) (range 1 m))))

; Check satisfiability
(check-sat)
