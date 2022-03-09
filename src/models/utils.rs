use std::ops::Range;

use grb::{Model, Result, Var, VarType};

pub trait AddVars {
    type Out;

    /// Create a variable with a closure
    fn vars_with<F: FnMut(Self) -> Result<Var>>(&self, func: F) -> Result<Self::Out>
    where
        Self: Sized;

    /// Create a variable for any type
    fn vars(
        &self,
        model: &mut Model,
        base_name: &str,
        vtype: VarType,
        bounds: &Range<f64>,
    ) -> Result<Self::Out>;

    /// Binary variables
    fn binary(&self, model: &mut Model, base_name: &str) -> Result<Self::Out> {
        self.vars(
            model,
            base_name,
            VarType::Binary,
            &(f64::NEG_INFINITY..f64::INFINITY),
        )
    }

    /// A continuous non-negative variable
    fn cont(&self, model: &mut Model, base_name: &str) -> Result<Self::Out> {
        self.vars(model, base_name, VarType::Continuous, &(0.0..f64::INFINITY))
    }

    /// A free continuous variable
    fn free(&self, model: &mut Model, base_name: &str) -> Result<Self::Out> {
        self.vars(
            model,
            base_name,
            VarType::Continuous,
            &(f64::NEG_INFINITY..f64::INFINITY),
        )
    }
}

impl AddVars for usize {
    type Out = Vec<Var>;

    fn vars_with<F: FnMut(Self) -> Result<Var>>(&self, mut func: F) -> Result<Self::Out>
    where
        Self: Sized,
    {
        let mut vec = Vec::with_capacity(*self);
        for i in 0..*self {
            vec.push(func(i)?);
        }

        Ok(vec)
    }

    fn vars(
        &self,
        model: &mut Model,
        base_name: &str,
        vtype: VarType,
        bounds: &Range<f64>,
    ) -> Result<Self::Out> {
        let mut vec = Vec::with_capacity(*self);
        for i in 0..*self {
            vec.push(model.add_var(
                &format!("{}_{}", base_name, i),
                vtype,
                0.0,
                bounds.start,
                bounds.end,
                std::iter::empty(),
            )?);
        }

        Ok(vec)
    }
}

impl AddVars for (usize, usize) {
    type Out = Vec<<usize as AddVars>::Out>;
    fn vars(
        &self,
        model: &mut Model,
        base_name: &str,
        vtype: VarType,
        bounds: &Range<f64>,
    ) -> Result<Self::Out> {
        let mut out = Vec::with_capacity(self.0);
        for i in 0..self.0 {
            out.push(
                self.1
                    .vars(model, &format!("{}_{}", base_name, i), vtype, bounds)?,
            )
        }

        Ok(out)
    }

    fn vars_with<F: FnMut(Self) -> Result<Var>>(&self, mut func: F) -> Result<Self::Out>
    where
        Self: Sized,
    {
        let mut out = Vec::with_capacity(self.0);
        for i in 0..self.0 {
            out.push(self.0.vars_with(|j| func((i, j)))?);
        }

        Ok(out)
    }
}

impl AddVars for (usize, usize, usize) {
    type Out = Vec<<(usize, usize) as AddVars>::Out>;
    fn vars(
        &self,
        model: &mut Model,
        base_name: &str,
        vtype: VarType,
        bounds: &Range<f64>,
    ) -> Result<Self::Out> {
        let mut out = Vec::with_capacity(self.0);
        for i in 0..self.0 {
            out.push((self.1, self.2).vars(
                model,
                &format!("{}_{}", base_name, i),
                vtype,
                bounds,
            )?)
        }

        Ok(out)
    }

    fn vars_with<F: FnMut(Self) -> Result<Var>>(&self, mut func: F) -> Result<Self::Out>
    where
        Self: Sized,
    {
        let mut out = Vec::with_capacity(self.0);
        for i in 0..self.0 {
            out.push((self.1, self.2).vars_with(|(j, k)| func((i, j, k)))?)
        }

        Ok(out)
    }
}

impl AddVars for (usize, usize, usize, usize) {
    type Out = Vec<<(usize, usize, usize) as AddVars>::Out>;
    fn vars(
        &self,
        model: &mut Model,
        base_name: &str,
        vtype: VarType,
        bounds: &Range<f64>,
    ) -> Result<Self::Out> {
        let mut out = Vec::with_capacity(self.0);
        for i in 0..self.0 {
            out.push((self.1, self.2, self.3).vars(
                model,
                &format!("{}_{}", base_name, i),
                vtype,
                bounds,
            )?)
        }

        Ok(out)
    }

    fn vars_with<F: FnMut(Self) -> Result<Var>>(&self, mut func: F) -> Result<Self::Out>
    where
        Self: Sized,
    {
        let mut out = Vec::with_capacity(self.0);
        for i in 0..self.0 {
            out.push((self.1, self.2, self.3).vars_with(|(j, k, l)| func((i, j, k, l)))?)
        }

        Ok(out)
    }
}

impl AddVars for (usize, usize, usize, usize, usize) {
    type Out = Vec<<(usize, usize, usize, usize) as AddVars>::Out>;
    fn vars(
        &self,
        model: &mut Model,
        base_name: &str,
        vtype: VarType,
        bounds: &Range<f64>,
    ) -> Result<Self::Out> {
        let mut out = Vec::with_capacity(self.0);
        for i in 0..self.0 {
            out.push((self.1, self.2, self.3, self.4).vars(
                model,
                &format!("{}_{}", base_name, i),
                vtype,
                bounds,
            )?)
        }

        Ok(out)
    }

    fn vars_with<F: FnMut(Self) -> Result<Var>>(&self, mut func: F) -> Result<Self::Out>
    where
        Self: Sized,
    {
        let mut out = Vec::with_capacity(self.0);
        for i in 0..self.0 {
            out.push(
                (self.1, self.2, self.3, self.4).vars_with(|(j, k, l, m)| func((i, j, k, l, m)))?,
            )
        }

        Ok(out)
    }
}
