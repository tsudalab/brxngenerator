from __future__ import absolute_import, print_function, division

from collections import MutableSet
import types
import weakref

from six import string_types


def check_deterministic(iterable):
    if iterable is not None:
        if not isinstance(iterable, (
                list, tuple, OrderedSet,
                types.GeneratorType, string_types)):
            if len(iterable) > 1:
                raise AssertionError(
                    "Get an not ordered iterable when one was expected")





class Link(object):
    __slots__ = 'prev', 'next', 'key', '__weakref__'

    def __getstate__(self):
        ret = [self.prev(), self.next()]
        try:
            ret.append(self.key)
        except AttributeError:
            pass
        return ret

    def __setstate__(self, state):
        self.prev = weakref.ref(state[0])
        self.next = weakref.ref(state[1])
        if len(state) == 3:
            self.key = state[2]


class OrderedSet(MutableSet):
    'Set the remembers the order elements were added'

    def update(self, iterable):
        check_deterministic(iterable)
        self |= iterable

    def __init__(self, iterable=None):
        check_deterministic(iterable)
        self.__root = root = Link()         # sentinel node for doubly linked list
        root.prev = root.next = weakref.ref(root)
        self.__map = {}                     # key --> link
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.__map)

    def __contains__(self, key):
        return key in self.__map

    def add(self, key):
        if key not in self.__map:
            self.__map[key] = link = Link()
            root = self.__root
            last = root.prev
            link.prev, link.next, link.key = last, weakref.ref(root), key
            last().next = root.prev = weakref.ref(link)

    def union(self, s):
        check_deterministic(s)
        n = self.copy()
        for elem in s:
            if elem not in n:
                n.add(elem)
        return n

    def intersection_update(self, s):
        l = []
        for elem in self:
            if elem not in s:
                l.append(elem)
        for elem in l:
            self.remove(elem)
        return self

    def difference_update(self, s):
        check_deterministic(s)
        for elem in s:
            if elem in self:
                self.remove(elem)
        return self

    def copy(self):
        n = OrderedSet()
        n.update(self)
        return n

    def discard(self, key):
        if key in self.__map:
            link = self.__map.pop(key)
            link.prev().next = link.next
            link.next().prev = link.prev

    def __iter__(self):
        root = self.__root
        curr = root.next()
        while curr is not root:
            yield curr.key
            curr = curr.next()

    def __reversed__(self):
        root = self.__root
        curr = root.prev()
        while curr is not root:
            yield curr.key
            curr = curr.prev()

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        if last:
            key = next(reversed(self))
        else:
            key = next(iter(self))
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        elif isinstance(other, set):
            raise TypeError(
                'Cannot compare an `OrderedSet` to a `set` because '
                'this comparison cannot be made symmetric: please '
                'manually cast your `OrderedSet` into `set` before '
                'performing this comparison.')
        else:
            return NotImplemented


if __name__ == '__main__':
    print(list(OrderedSet('abracadaba')))
    print(list(OrderedSet('simsalabim')))
    print(OrderedSet('boom') == OrderedSet('moob'))
    print(OrderedSet('boom') == 'moob')
