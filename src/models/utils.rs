use grb::prelude::*;
use grb::{Expr, Model, Result, Var, VarType};
use std::hash::Hash;
use std::{collections::HashMap, fmt::Debug, ops::Range};

use super::exact_model::sets_and_parameters::{Arc, NetworkNode};

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
                &format!("{}{}", base_name, i),
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

impl AddVars for (&Vec<NetworkNode>, usize) {
    type Out = Vec<<usize as AddVars>::Out>;

    fn vars_with<F: FnMut(Self) -> Result<Var>>(&self, _func: F) -> Result<Self::Out>
    where
        Self: Sized,
    {
        todo!()
    }

    fn vars(
        &self,
        model: &mut Model,
        base_name: &str,
        vtype: VarType,
        bounds: &Range<f64>,
    ) -> Result<Self::Out> {
        let mut out = Vec::with_capacity(self.0.len());

        for node in self.0 {
            let base_name = match node.kind() {
                super::exact_model::sets_and_parameters::NetworkNodeType::Source => {
                    format!("{}({},{}),", base_name, -1, -1,)
                }
                super::exact_model::sets_and_parameters::NetworkNodeType::Sink => {
                    format!("{}({},{}),", base_name, -1, node.time(),)
                }
                super::exact_model::sets_and_parameters::NetworkNodeType::Normal => {
                    format!("{}({},{}),", base_name, node.port(), node.time(),)
                }
            };
            out.push(self.1.vars(model, &base_name, vtype, bounds)?)
        }

        Ok(out)
    }
}

impl AddVars for (&Vec<Arc>, usize) {
    type Out = Vec<<usize as AddVars>::Out>;

    fn vars_with<F: FnMut(Self) -> Result<Var>>(&self, _func: F) -> Result<Self::Out>
    where
        Self: Sized,
    {
        todo!()
    }

    fn vars(
        &self,
        model: &mut Model,
        base_name: &str,
        vtype: VarType,
        bounds: &Range<f64>,
    ) -> Result<Self::Out> {
        let mut out = Vec::with_capacity(self.0.len());

        for arc in self.0 {
            let base_name = match arc.get_kind() {
                super::exact_model::sets_and_parameters::ArcType::TravelArc => format!(
                    "{}({},{}),({},{}),",
                    base_name,
                    arc.get_from().port(),
                    arc.get_from().time(),
                    arc.get_to().port(),
                    arc.get_to().time(),
                ),
                super::exact_model::sets_and_parameters::ArcType::WaitingArc => format!(
                    "{}({},{}),({},{}),",
                    base_name,
                    arc.get_from().port(),
                    arc.get_from().time(),
                    arc.get_to().port(),
                    arc.get_to().time(),
                ),
                super::exact_model::sets_and_parameters::ArcType::EnteringArc => format!(
                    "{}({},{}),({},{}),",
                    base_name,
                    -1,
                    -1,
                    arc.get_to().port(),
                    arc.get_to().time(),
                ),
                super::exact_model::sets_and_parameters::ArcType::LeavingArc => format!(
                    "{}({},{}),({},{}),",
                    base_name,
                    arc.get_from().port(),
                    arc.get_from().time(),
                    -1,
                    arc.get_to().time(),
                ),
                super::exact_model::sets_and_parameters::ArcType::NotUsedArc => format!(
                    "{}({},{}),({},{}),",
                    base_name,
                    -1,
                    -1,
                    -1,
                    arc.get_to().time(),
                ),
            };
            out.push(self.1.vars(model, &base_name, vtype, bounds)?)
        }

        Ok(out)
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
                    .vars(model, &format!("{}{},", base_name, i), vtype, bounds)?,
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
                &format!("{}{},", base_name, i),
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
                &format!("{}{},", base_name, i),
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
                &format!("{}{},", base_name, i),
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

impl<K, V: ConvertVars> ConvertVars for HashMap<K, V>
where
    K: std::cmp::Eq + Hash + Clone,
{
    type Out = HashMap<K, V::Out>;

    fn convert(&self, model: &Model) -> grb::Result<Self::Out> {
        let mut out = HashMap::with_capacity(self.len());
        for (k, v) in self.into_iter() {
            out.insert(k.clone(), v.convert(model)?);
        }
        Ok(out)
    }
}

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

pub fn better_vars<K>(
    indices: Vec<K>,
    model: &mut Model,
    vtype: VarType,
    bounds: &Range<f64>,
    name: &str,
) -> Result<HashMap<K, Var>>
where
    K: Eq + Hash + Debug,
{
    let mut res = HashMap::new();
    for idx in indices {
        let var = model.add_var(
            &format!("{:?}_{:?}", name, idx),
            vtype,
            0.0,
            bounds.start,
            bounds.end,
            std::iter::empty(),
        )?;
        res.insert(idx, var);
    }
    Ok(res)
}
