#[derive(Default, Debug, Eq, Copy, Clone)]
pub struct bool32 {
    inner: u32,
}

unsafe impl ocl::traits::OclPrm for bool32 {
}

impl bool32 {
    pub fn t() -> bool32 {
        bool32 {
            inner: 1,
        }
    }

    pub fn f() -> bool32 {
        bool32 {
            inner: 0,
        }
    }
}

impl PartialEq for bool32 {
    fn eq(
        &self,
        other: &Self,
    ) -> bool
    {
        let this: bool = (*self).into();
        let that: bool = (*other).into();

        this == that
    }
}

impl Into<bool> for bool32 {
    fn into(self) -> bool {
        self.inner != 0
    }
}

impl From<bool> for bool32 {
    fn from(b: bool) -> bool32 {
        if b {
            bool32::t()
        }
        else {
            bool32::f()
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

pub struct ForceTruncOrZeroRepeatIter<I, T>
where
    I: Iterator<Item = T>,
    T: Default, {
    iter:  I,
    count: usize,
}

impl<I, T> ForceTruncOrZeroRepeatIter<I, T>
where
    I: Iterator<Item = T>,
    T: Default,
{
    pub fn new(
        iter: I,
        count: usize,
    ) -> ForceTruncOrZeroRepeatIter<I, T>
    {
        ForceTruncOrZeroRepeatIter {
            iter,
            count,
        }
    }
}

impl<I, T> Iterator for ForceTruncOrZeroRepeatIter<I, T>
where
    I: Iterator<Item = T>,
    T: Default,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }

        self.count -= 1;

        self.iter.next().or_else(|| Some(Default::default()))
    }
}

impl<I, T> ExactSizeIterator for ForceTruncOrZeroRepeatIter<I, T>
where
    I: Iterator<Item = T>,
    T: Default,
{
    fn len(&self) -> usize {
        self.count
    }
}
