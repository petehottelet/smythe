//! Exact nano-USD values and identity-bound reservations.
use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct NanoUsd(pub u64);

impl Serialize for NanoUsd {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.to_string())
    }
}
impl<'de> Deserialize<'de> for NanoUsd {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        if value.is_empty()
            || !value.bytes().all(|b| b.is_ascii_digit())
            || (value.len() > 1 && value.starts_with('0'))
        {
            return Err(serde::de::Error::custom(
                "money requires a canonical unsigned decimal string",
            ));
        }
        value.parse().map(Self).map_err(serde::de::Error::custom)
    }
}
impl NanoUsd {
    pub fn checked_add(self, other: Self) -> Result<Self> {
        self.0
            .checked_add(other.0)
            .map(Self)
            .ok_or_else(|| anyhow!("money overflow"))
    }
    /// Conservative quote; pricing is nano-USD per million units.
    pub fn quote(units: u64, rate_per_million: u64) -> Result<Self> {
        let product = u128::from(units) * u128::from(rate_per_million);
        u64::try_from(product.div_ceil(1_000_000))
            .map(Self)
            .map_err(Into::into)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sentinel {
    pub ceiling: NanoUsd,
    pub spent: NanoUsd,
    reservations: BTreeMap<String, NanoUsd>,
    settlements: BTreeMap<String, NanoUsd>,
}
impl Sentinel {
    pub fn new(ceiling: NanoUsd) -> Self {
        Self {
            ceiling,
            spent: NanoUsd(0),
            reservations: BTreeMap::new(),
            settlements: BTreeMap::new(),
        }
    }
    pub fn exposure(&self) -> Result<NanoUsd> {
        self.reservations
            .values()
            .try_fold(self.spent, |sum, &amount| sum.checked_add(amount))
    }
    pub fn reserve(&mut self, key: &str, amount: NanoUsd) -> Result<()> {
        ensure!(
            !key.is_empty()
                && !self.reservations.contains_key(key)
                && !self.settlements.contains_key(key),
            "duplicate reservation identity"
        );
        ensure!(
            self.exposure()?.checked_add(amount)? <= self.ceiling,
            "budget exhausted before dispatch"
        );
        self.reservations.insert(key.to_owned(), amount);
        Ok(())
    }
    /// Returns whether the quote was exceeded, while retaining all known spend.
    /// Replaying identical evidence is harmless; conflicting evidence is refused.
    pub fn settle(&mut self, key: &str, actual: NanoUsd) -> Result<bool> {
        if let Some(previous) = self.settlements.get(key) {
            ensure!(*previous == actual, "conflicting settlement evidence");
            return Ok(false);
        }
        let bound = *self
            .reservations
            .get(key)
            .ok_or_else(|| anyhow!("unknown reservation"))?;
        let spent = self.spent.checked_add(actual)?;
        self.spent = spent;
        self.reservations.remove(key);
        self.settlements.insert(key.into(), actual);
        Ok(actual > bound
            || self
                .exposure()
                .map_or(true, |exposure| exposure > self.ceiling))
    }
    pub fn pending(&self) -> &BTreeMap<String, NanoUsd> {
        &self.reservations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn money_round_trips_above_javascript_precision_without_floats() {
        let amount = NanoUsd(u64::MAX);
        let encoded = serde_json::to_string(&amount).unwrap();
        assert_eq!(encoded, "\"18446744073709551615\"");
        assert_eq!(serde_json::from_str::<NanoUsd>(&encoded).unwrap(), amount);
        for bad in ["1", "1.0", "\"-1\"", "\"01\"", "\"18446744073709551616\""] {
            assert!(serde_json::from_str::<NanoUsd>(bad).is_err());
        }
        assert_eq!(NanoUsd::quote(1, 1).unwrap(), NanoUsd(1));
        assert!(NanoUsd::quote(u64::MAX, u64::MAX).is_err());
    }
    #[test]
    fn settlement_retains_overruns_and_is_idempotent() {
        let mut s = Sentinel::new(NanoUsd(10));
        s.reserve("a", NanoUsd(4)).unwrap();
        s.reserve("b", NanoUsd(6)).unwrap();
        assert!(s.settle("a", NanoUsd(7)).unwrap());
        assert_eq!(s.spent, NanoUsd(7));
        assert_eq!(s.exposure().unwrap(), NanoUsd(13));
        assert!(s.reserve("c", NanoUsd(0)).is_err());
        s.settle("a", NanoUsd(7)).unwrap();
        assert!(s.settle("a", NanoUsd(8)).is_err());
        assert!(s.pending().contains_key("b"));
    }
    #[test]
    fn rejected_reservation_and_overflow_leave_ledger_unchanged() {
        let mut s = Sentinel::new(NanoUsd(u64::MAX));
        s.reserve("a", NanoUsd(u64::MAX)).unwrap();
        assert!(s.reserve("a", NanoUsd(0)).is_err());
        assert!(s.reserve("b", NanoUsd(1)).is_err());
        s.settle("a", NanoUsd(u64::MAX)).unwrap();
        assert_eq!(s.spent, NanoUsd(u64::MAX));
    }
    #[test]
    fn exposure_overflow_keeps_known_spend_and_closes_admission() {
        let mut s = Sentinel::new(NanoUsd(u64::MAX));
        s.reserve("a", NanoUsd(1)).unwrap();
        s.reserve("b", NanoUsd(1)).unwrap();
        assert!(s.settle("a", NanoUsd(u64::MAX)).unwrap());
        assert_eq!(s.spent, NanoUsd(u64::MAX));
        assert!(s.pending().contains_key("b"));
        assert!(s.reserve("c", NanoUsd(0)).is_err());
    }
}
