from sympy.matrices import Matrix, eye, zeros


def stabilize_ranks(A, eigenval):
  '''
  Given square matrix A and an eigen value of it
  compute rank(A-(eigenval)I)^p for each p=0,1,2,...
  keeps track of ranks for each p
  stop once ranks stabilize.

  A: square matrix (sympy)

  eigenval: one eigenvalue of A (int)

  return r: record of ranks until stabilization (list<int>)
  '''

  r = []
  p = 0

  while True:
    rank = ((A-eigenval*eye(A.shape[0]))**p).rank()
    r.append(rank)

    if rank == r[p-1] and p != 0:
      return r
    p +=1


def weyr(r):
  '''
  Given the record of ranks until stabilization for (A-eigenval*I) "r",
  compute weyr characteristic.

  r: record of ranks until stabilization (list<int>)

  return w: weyr characteristic (list<int>)
  '''
  w = []
  for p in range(1, len(r)):
    w_p = r[p-1] - r[p]
    w.append(w_p)

  w.append(0) # NOT PART OF THE WEYR CHARACTERISTIC, FOR JORDAN FORM COMPUTATION PURPOSE ONLY
  return w

def concat_vec(v):
  '''
  For a list of sympy vectors (nX1 matrices) "v", combine them into
  a sympy matrix where the index of the vector corresponds to its
  matrix column.

  v: list of nX1 matrices (sympy)

  return M: matrix of given vectors (sympy)
  '''
  M = v[0]

  for i in range(1,len(v)):
    M = M.row_join(v[i])

  return M

def get_super(n):
  '''
  For size "n" construct the nXn matrix with a
  super diagonal composed of 1's.

  n: size of square matrix (int)

  return M: size n square matrix with super diagonal of 1's (sympy)
  '''
  v = []
  v_i = [0]*n
  v.append(Matrix((v_i)))

  for i in range(0,n-1):
    v_i = [0]*n
    v_i[i]=1
    v.append(Matrix((v_i)))

  M = concat_vec(v)

  return M

def direct_summand(A, B):
  '''
  For two sympy matrices "A" and "B", compute the direct summand
  this being matrices "A" and "B" on the diagonal and
  zero matrices on the anti-diagonal.

  A: Matrix (sympy)

  B: Matrix (sympy)

  return M: direct summand of A and B (sympy)
  '''
  lower_zero = zeros(B.shape[0],A.shape[1])
  upper_zero = zeros(A.shape[1],B.shape[0])

  upper = A.row_join(upper_zero)
  lower = lower_zero.row_join(B)

  return upper.col_join(lower)

def many_summand(ml):
  '''
  For a list of matrices, compute the direct summand of
  all matrices in the list.

  ml: list of matrices (list<sympy>)
  '''
  if len(ml) == 1:
    return ml[0]

  if len(ml) > 1:
    M = direct_summand(ml[0], ml[1])

    for i in range(2, len(ml)):
      M = direct_summand(M, ml[i])

    return M

  else:
     return

def jordan_block(eigenval, n):
  '''
  Given an egienvalue and matrix size "n", create an
  nXn jordan block with with the given eigenvalue.

  egienval: eigenvalue (int)

  n: size of jordan block (int)
  '''
  block = eigenval*eye(n)+get_super(n)

  return(block)

def create_blocks(weyr, eigenval):
  '''
  For a given eigenvalue and its corresponding "weyr" characteristic
  first compute the differences between each element of the "weyr"
  characteristic and their following element, then store in "block_info".
  For each value in "block_info", if the value is greater than zero
  then create that value of jordan blocks of size corresponding to the
  index of the value and append to blocks. Finally, compute direct summand
  all blocks created.

  weyr: weyr characteristic of some matrix A corresponding to an eigenvalue (list<int>)

  eigenval: corresponding eigenvalue of A to the weyr characteristic (int)

  return M: the direct summand of all jordan blocks created (sympy)
  '''
  block_info= []

  for i in range(1,len(weyr)):
    bi = weyr[i-1]-weyr[i]
    block_info.append(bi)

  blocks = []
  for i in reversed(range(len(block_info))): # reverse is to order blocks accourding to size, this is stylistic only
    if block_info[i] > 0:
      for _ in range(block_info[i]):
        block = jordan_block(eigenval,i+1)
        blocks.append(block)

  M = many_summand(blocks)

  return M

def jordan_form(A):
  '''
  compute the jordan form of sympy matrix "A". First get all eigenvalues of "A".
  Then for each eigenvalue of "A" compute the corresponding weyr characteristic.
  After computing the weyr characteristic compute the Jordan blocks that 
  correspond to it and append to "blocks". Finally, compute direct summand of 
  all blocks found to create the jordan form of "A".

  A: square matrix (sympy)

  return M: jordan form of A (sympy)
  '''
  eigenvals = A.eigenvals()
  vals = []
  for key in eigenvals:
    vals.append(key)

  blocks = []
  for e in vals:

    r = stabilize_ranks(A,e)
    w = weyr(r)

    e_m=create_blocks(w, e)
    blocks.append(e_m)

  if len(blocks) > 1:
    return many_summand(blocks)

  else:
    return blocks[0]
