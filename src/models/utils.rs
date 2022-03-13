use grb::{Expr, Model, Result, Var, VarType};
use std::ops::Range;

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
            out.push(self.1.vars_with(|j| func((i, j)))?);
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

use grb::prelude::*;
#[allow(non_snake_case)]
pub trait NObjectives {
    /// Adds a new objective to the model
    ///
    /// # Arugments
    ///
    /// *`expr` - An expression of type `Expr`, must be linear
    ///
    /// *`priority` - The priority of the objective. Lowest priority is assigned the value 0.
    ///
    /// *`index` - The index of the objective, i.e. objective numer *i* in the model. **Starts at 0.**
    ///
    /// *`name` - A string slice that holds the name of the constraint.
    fn set_objective_N(
        &mut self,
        expr: impl Into<Expr>,
        priority: i32,
        index: i32,
        name: &str,
    ) -> grb::Result<()>;
}

impl NObjectives for Model {
    fn set_objective_N(
        &mut self,
        expr: impl Into<Expr>,
        priority: i32,
        index: i32,
        name: &str,
    ) -> grb::Result<()> {
        self.update()?;
        let expr: Expr = expr.into();
        let expr = if expr.is_linear() {
            expr.into_linexpr()
        } else {
            let error = grb::Error::AlgebraicError(
                format!("Tried to add multiple objectives where at least one is non linear\nNon linear objective:{:?}",expr),
            );
            Err(error)
        }?;

        let (coeff_map, obj_cons) = expr.into_parts();

        // number of objective is set to the max of the current number and the index+1
        let num_objectives = i32::max(self.get_attr(attr::NumObj)?, index + 1);
        self.set_attr(attr::NumObj, num_objectives)?;

        // set the objNumber which is the number of the object that is currently worked with
        self.set_param(param::ObjNumber, index)?;
        self.set_obj_attr_batch(attr::ObjN, coeff_map)?;
        self.set_attr(attr::ObjNCon, obj_cons)?;
        self.set_attr(attr::ObjNPriority, priority)?;
        self.set_attr(attr::ObjNName, name)?;

        Ok(())
    }
}

/// Trait that converts gurobi varaibles to f64
pub trait ConvertVars {
    type Out;
    fn convert(&self, model: &Model) -> grb::Result<Self::Out>;
}

/* impl<T> ConvertVars<T> for Vec<T>
where
    T: ConvertVars<T>,
{
    type Out = Vec<<T as ConvertVars<T>>::Out>;

    fn convert(&self, model: &Model) -> grb::Result<Self::Out> {
        let mut out = Vec::with_capacity(self.len());
        for e in self {
            out.push(e.convert(model)?);
        }
        Ok(out)
    }
} */

impl<T: ConvertVars> ConvertVars for Vec<T> {
    type Out = Vec<T::Out>;

    fn convert(&self, model: &Model) -> grb::Result<Self::Out> {
        let mut out = Vec::with_capacity(self.len());
        for e in self {
            out.push(e.convert(model)?);
        }
        Ok(out)
    }
}

impl ConvertVars for Var {
    type Out = f64;

    fn convert(&self, model: &Model) -> grb::Result<Self::Out> {
        model.get_obj_attr(attr::X, self)
    }
}
