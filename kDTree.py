import numpy


class node(object):
    """ Node of kD-Tree """
    def __init__(self, dim, value, label, parent):
        #dim is the dimension along which the tree is partitioned at
        #this node
        self.dim = dim
        self.value = value
        self.parent = parent
        self.label = label
        self.is_leaf = False

    def set_children(self, child1, child2):
        self.left = child1
        self.right = child2

    def sqdistance(self, vector):
        return (numpy.linalg.norm(vector - self.value))**2

    def __str__(self):
        return str(self.value)


class kDTree(object):
    """
    Class to represent kD-Trees in k-dimensional space
    Built to perform k-NN searches
    """
    
    def __init__(self, vectors, labels):
        """
        Initializer
        Builds the kDtree
        """
        if len(vectors) != len(labels):
            raise ValueError("Length of vector and label lists must be same")
        self._number = len(vectors)
        self.k = len(vectors[0])
        if len(vectors) == 0:
            raise ValueError("Empty vector list supplied")
        lookup = {}
        for i, x in enumerate(vectors):
            lookup[tuple(x)] = labels[i]
        self._root = self._build_tree(vectors, None, 0, lookup)

    def _build_tree(self, vectors, parent, dim, lookup):
        #Special case of leaf
        if len(vectors) == 1:
            root = node(dim, vectors[0], lookup[tuple(vectors[0])], parent)
            root.set_children(None, None)
            root.is_leaf = True
            return root
        #Else, build tree recursively
        flag = False
        if len(set([tuple(x) for x in vectors])) == 1:
            flag = True
        next_dim = self._next_dim(dim)
        vectors.sort(key = lambda x: x[dim])
        median = vectors[int(len(vectors)/2)]
        i = int(len(vectors)/2)
        while vectors[i][dim] == median[dim] and i > -1:
            i -= 1
        if flag:
            i = int(len(vectors)/2) - 1
        median = vectors[i+1]
        setr = vectors[i+2:]
        setl = vectors[:i+1]
        root = node(dim, median, lookup[tuple(median)], parent)
        if len(setr) == 0:
            right = None
        else:
            right = self._build_tree(setr, root, next_dim, lookup)
        if len(setl) == 0:
            left = None
        else:
            left = self._build_tree(setl, root, next_dim, lookup)
        root.set_children(left, right)
        return root

    def _next_dim(self, last):
        if last >= self.k - 1:
            return 0
        return last+1

    def nearest_neighbours(self, vector, n):
        """
        Returns the approx. n nearest neighbours to given vector and also their
        respective distances from the given vector
        """

        if n > self._number / 2:
            raise ValueError("Value of n is too high wrt dataset; use iterative \
                            search instead")
        if self._number == 1:
            return self._root.value
        node = self._reach_leaf(self._root, vector)
        kNN = [None for i in range(n)]
        kdists = [-1 for i in range(n)]
        currentdist = -1
        currentdist, kNN, kdists = self._NN_helper(node, vector, kNN,
                                                   kdists, currentdist)
        kNN = [x.label for x in kNN]
        kdists = [x ** 0.5 for x in kdists]
        return kNN, kdists

    def _NN_helper(self, node, vector, kNN, kdists, currentdist, direct=None):
        #If node is None, do nothing.
        if node is None:
            return currentdist, kNN, kdists
        #If currentdist = -1, then k neighbours have not yet been encountered
        #(excluding this one)
        nv_dist = node.sqdistance(vector)
        if currentdist == -1:
            i = kNN.index(None)
            kNN[i] = node
            kdists[i] = nv_dist
            #Check if k neighbours have now been encountered including this one
            #If yes, update currentdist accordingly
            if -1 not in kdists:
                currentdist = max(kdists)
            #Else, currentdist remains -1
        else:
            #k neighbours have been encountered
            #Check if current node is within sphere of k-distance
            if currentdist > nv_dist:
                #If yes, remove current farthest node, and include this one
                #Then set currentdist accordingly
                i = kdists.index(currentdist)
                kdists[i] = nv_dist
                kNN[i] = node
                currentdist = max(kdists)
        #If direct is None, it means we are at beginning of process
        #else, check if currentdist is greater than difference between values of node
        #and vector on splitting dimension
        if direct is not None and \
           (currentdist > abs(node.value[node.dim] - vector[node.dim]) or
            currentdist == -1):
            #If yes, explore the other side of the hyperplane
            if direct == 'l':
                if node.right is not None:
                    node.right.parent = None
                    node_to_process = self._reach_leaf(node.right, vector)
                    currentdist, kNN, kdists = self._NN_helper(node_to_process, vector,
                                                               kNN, kdists,
                                                               currentdist)
                    node.right.parent = node
            else:
                if node.left is not None:
                    node.left.parent = None
                    node_to_process = self._reach_leaf(node.left, vector)
                    currentdist, kNN, kdists = self._NN_helper(node_to_process, vector,
                                                               kNN, kdists,
                                                               currentdist)
                    node.left.parent = node
        #If parent is None, we are at the top of the tree. Return.
        if node.parent is None:
            return currentdist, kNN, kdists
        #Else, send the recursion to the parent
        if node == node.parent.right:
            return self._NN_helper(node.parent, vector, kNN, kdists, currentdist, 'r')
        else:
            return self._NN_helper(node.parent, vector, kNN, kdists, currentdist, 'l')

    def _reach_leaf(self, root, vector):
        node = root
        while not node.is_leaf:
            if vector[node.dim] >= node.value[node.dim]:
                if node.right is None:
                    return node
                node = node.right
            else:
                if node.left is None:
                    return node
                node = node.left
        return node

class IterativeSearch(object):
    """
    Normal iterative search object
    """

    _temp_list = []

    def __init__(self, vectors, labels):
        """
        Initializer
        """

        if len(vectors) != len(labels):
            raise ValueError("Length of vector and label lists must be same")
        self._number = len(vectors)
        lookup = {}
        for i, x in enumerate(vectors):
            lookup(tuple(x)) = labels[i]
        self._vectors = vectors

    def _update_temp(self, vector):
        self._temp_list = []
        for x in self._vectors:
            self._temp_list.append(numpy.linalg.norm(x - vector))

    def nearest_neighbours(self, vector, n):
        if n > self._number:
            raise ValueError("Value of n is too high")
        self._update_temp(vector)
        temp = self._vectors[:]
        temp.sort(key = lambda x : _temp_list[self._vectors.index(x)])
        self._temp_list.sort
        o_labels = [lookup[tuple(x)] for x in temp[:n+1]]
        return o_labels, self._temp_list[:n+1]
        
        
