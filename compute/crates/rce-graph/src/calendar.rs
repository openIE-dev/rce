//! Calendar and date math: Julian day, day of week, leap year,
//! date arithmetic, epoch conversions, holiday detection, astronomical.

// ═══════════════════════════════════════════════════════════════════
// JULIAN DAY NUMBER
// ═══════════════════════════════════════════════════════════════════

/// Convert Gregorian date to Julian Day Number.
pub fn date_to_jdn(year: i32, month: u32, day: u32) -> i64 {
    let a = (14 - month as i64) / 12;
    let y = year as i64 + 4800 - a;
    let m = month as i64 + 12 * a - 3;
    day as i64 + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045
}

/// Convert Julian Day Number to Gregorian date (year, month, day).
pub fn jdn_to_date(jdn: i64) -> (i32, u32, u32) {
    let a = jdn + 32044;
    let b = (4 * a + 3) / 146097;
    let c = a - (146097 * b) / 4;
    let d = (4 * c + 3) / 1461;
    let e = c - (1461 * d) / 4;
    let m = (5 * e + 2) / 153;
    let day = (e - (153 * m + 2) / 5 + 1) as u32;
    let month = (m + 3 - 12 * (m / 10)) as u32;
    let year = (100 * b + d - 4800 + m / 10) as i32;
    (year, month, day)
}

/// Day of week (0=Monday, 6=Sunday) for a Gregorian date.
pub fn day_of_week(year: i32, month: u32, day: u32) -> u32 {
    let jdn = date_to_jdn(year, month, day);
    ((jdn % 7 + 7) % 7) as u32
}

/// Day of week name.
pub fn day_name(dow: u32) -> &'static str {
    match dow % 7 { 0 => "Monday", 1 => "Tuesday", 2 => "Wednesday", 3 => "Thursday",
                    4 => "Friday", 5 => "Saturday", _ => "Sunday" }
}

/// Month name.
pub fn month_name(month: u32) -> &'static str {
    match month { 1 => "January", 2 => "February", 3 => "March", 4 => "April",
                  5 => "May", 6 => "June", 7 => "July", 8 => "August",
                  9 => "September", 10 => "October", 11 => "November", _ => "December" }
}

// ═══════════════════════════════════════════════════════════════════
// LEAP YEAR
// ═══════════════════════════════════════════════════════════════════

/// Is the year a leap year (Gregorian)?
pub fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Days in a given month.
pub fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if is_leap_year(year) { 29 } else { 28 },
        _ => 0,
    }
}

/// Days in a given year.
pub fn days_in_year(year: i32) -> u32 {
    if is_leap_year(year) { 366 } else { 365 }
}

/// Day of year (1-366).
pub fn day_of_year(year: i32, month: u32, day: u32) -> u32 {
    let mut doy = 0;
    for m in 1..month { doy += days_in_month(year, m); }
    doy + day
}

/// ISO week number and year.
pub fn iso_week(year: i32, month: u32, day: u32) -> (i32, u32) {
    let doy = day_of_year(year, month, day) as i32;
    let dow = day_of_week(year, month, day) as i32;
    let dow_jan1 = day_of_week(year, 1, 1) as i32;
    let mut week = (doy + dow_jan1 - dow - 1) / 7 + 1;
    let mut iso_year = year;
    if week < 1 { iso_year -= 1; week = iso_weeks_in_year(iso_year) as i32; }
    else if week > iso_weeks_in_year(year) as i32 { iso_year += 1; week = 1; }
    (iso_year, week as u32)
}

/// Number of ISO weeks in a year.
pub fn iso_weeks_in_year(year: i32) -> u32 {
    let dow_jan1 = day_of_week(year, 1, 1);
    if dow_jan1 == 3 || (is_leap_year(year) && dow_jan1 == 2) { 53 } else { 52 }
}

// ═══════════════════════════════════════════════════════════════════
// DATE ARITHMETIC
// ═══════════════════════════════════════════════════════════════════

/// Add days to a date. Returns (year, month, day).
pub fn add_days(year: i32, month: u32, day: u32, n: i64) -> (i32, u32, u32) {
    jdn_to_date(date_to_jdn(year, month, day) + n)
}

/// Difference in days between two dates.
pub fn days_between(y1: i32, m1: u32, d1: u32, y2: i32, m2: u32, d2: u32) -> i64 {
    date_to_jdn(y2, m2, d2) - date_to_jdn(y1, m1, d1)
}

/// Add months to a date (clamp day to valid range).
pub fn add_months(year: i32, month: u32, day: u32, n: i32) -> (i32, u32, u32) {
    let total_months = (year * 12 + month as i32 - 1) + n;
    let new_year = total_months.div_euclid(12);
    let new_month = (total_months.rem_euclid(12) + 1) as u32;
    let max_day = days_in_month(new_year, new_month);
    (new_year, new_month, day.min(max_day))
}

/// Add years to a date.
pub fn add_years(year: i32, month: u32, day: u32, n: i32) -> (i32, u32, u32) {
    let new_year = year + n;
    let max_day = days_in_month(new_year, month);
    (new_year, month, day.min(max_day))
}

/// Age in years between birth date and reference date.
pub fn age_in_years(birth_y: i32, birth_m: u32, birth_d: u32, ref_y: i32, ref_m: u32, ref_d: u32) -> u32 {
    let mut age = (ref_y - birth_y) as u32;
    if ref_m < birth_m || (ref_m == birth_m && ref_d < birth_d) { age -= 1; }
    age
}

// ═══════════════════════════════════════════════════════════════════
// EPOCH CONVERSIONS
// ═══════════════════════════════════════════════════════════════════

/// Unix timestamp to Gregorian date (UTC). Returns (year, month, day, hour, min, sec).
pub fn unix_to_date(timestamp: i64) -> (i32, u32, u32, u32, u32, u32) {
    let days = timestamp.div_euclid(86400);
    let rem = timestamp.rem_euclid(86400) as u32;
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let sec = rem % 60;
    let jdn = days + 2440588; // Unix epoch JDN
    let (y, m, d) = jdn_to_date(jdn);
    (y, m, d, hour, min, sec)
}

/// Gregorian date (UTC) to Unix timestamp.
pub fn date_to_unix(year: i32, month: u32, day: u32, hour: u32, min: u32, sec: u32) -> i64 {
    let jdn = date_to_jdn(year, month, day);
    (jdn - 2440588) * 86400 + hour as i64 * 3600 + min as i64 * 60 + sec as i64
}

/// Modified Julian Date from Gregorian.
pub fn modified_julian_date(year: i32, month: u32, day: u32) -> f64 {
    date_to_jdn(year, month, day) as f64 - 2400000.5
}

// ═══════════════════════════════════════════════════════════════════
// BUSINESS DAYS
// ═══════════════════════════════════════════════════════════════════

/// Is date a weekday (Monday-Friday)?
pub fn is_weekday(year: i32, month: u32, day: u32) -> bool {
    day_of_week(year, month, day) < 5
}

/// Is date a weekend (Saturday or Sunday)?
pub fn is_weekend(year: i32, month: u32, day: u32) -> bool {
    day_of_week(year, month, day) >= 5
}

/// Count business days between two dates.
pub fn business_days_between(y1: i32, m1: u32, d1: u32, y2: i32, m2: u32, d2: u32) -> i64 {
    let total = days_between(y1, m1, d1, y2, m2, d2);
    let mut count = 0i64;
    let sign = if total >= 0 { 1 } else { -1 };
    for i in 1..=total.abs() {
        let (y, m, d) = add_days(y1, m1, d1, i * sign);
        if is_weekday(y, m, d) { count += 1; }
    }
    count * sign
}

/// Add n business days to a date.
pub fn add_business_days(year: i32, month: u32, day: u32, n: i32) -> (i32, u32, u32) {
    let mut remaining = n.abs();
    let step: i64 = if n >= 0 { 1 } else { -1 };
    let mut jdn = date_to_jdn(year, month, day);
    while remaining > 0 {
        jdn += step;
        let (y, m, d) = jdn_to_date(jdn);
        if is_weekday(y, m, d) { remaining -= 1; }
    }
    jdn_to_date(jdn)
}

// ═══════════════════════════════════════════════════════════════════
// EASTER & HOLIDAYS
// ═══════════════════════════════════════════════════════════════════

/// Easter Sunday date for a given year (Anonymous Gregorian algorithm).
pub fn easter_date(year: i32) -> (u32, u32) {
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;
    (month as u32, day as u32)
}

/// Check if date is a common US holiday (simplified).
pub fn is_us_holiday(year: i32, month: u32, day: u32) -> bool {
    // Fixed holidays
    if (month == 1 && day == 1) || // New Year's
       (month == 7 && day == 4) || // Independence Day
       (month == 12 && day == 25) || // Christmas
       (month == 6 && day == 19) || // Juneteenth
       (month == 11 && day == 11) { // Veterans Day
        return true;
    }
    // MLK Day: 3rd Monday in January
    if month == 1 && day_of_week(year, month, day) == 0 && day >= 15 && day <= 21 { return true; }
    // Presidents' Day: 3rd Monday in February
    if month == 2 && day_of_week(year, month, day) == 0 && day >= 15 && day <= 21 { return true; }
    // Memorial Day: last Monday in May
    if month == 5 && day_of_week(year, month, day) == 0 && day >= 25 { return true; }
    // Labor Day: 1st Monday in September
    if month == 9 && day_of_week(year, month, day) == 0 && day <= 7 { return true; }
    // Thanksgiving: 4th Thursday in November
    if month == 11 && day_of_week(year, month, day) == 3 && day >= 22 && day <= 28 { return true; }
    false
}

// ═══════════════════════════════════════════════════════════════════
// ASTRONOMICAL
// ═══════════════════════════════════════════════════════════════════

/// Approximate Julian century from J2000.0.
pub fn julian_century(year: i32, month: u32, day: u32) -> f64 {
    let jd = date_to_jdn(year, month, day) as f64;
    (jd - 2451545.0) / 36525.0
}

/// Approximate equation of time (minutes) for a given day of year.
pub fn equation_of_time(day_of_yr: f64) -> f64 {
    let b = 2.0 * std::f64::consts::PI / 365.0 * (day_of_yr - 81.0);
    9.87 * (2.0 * b).sin() - 7.53 * b.cos() - 1.5 * b.sin()
}

/// Solar declination (degrees) for a given day of year.
pub fn solar_declination(day_of_yr: f64) -> f64 {
    23.45 * (2.0 * std::f64::consts::PI / 365.0 * (day_of_yr - 81.0)).sin()
}

/// Approximate sunrise/sunset hour angle (degrees) for latitude and declination.
pub fn hour_angle_sunrise(latitude_deg: f64, declination_deg: f64) -> f64 {
    let lat = latitude_deg.to_radians();
    let dec = declination_deg.to_radians();
    let cos_h = -(lat.tan() * dec.tan());
    if cos_h > 1.0 { return 0.0; } // no sunrise (polar night)
    if cos_h < -1.0 { return 180.0; } // no sunset (midnight sun)
    cos_h.acos().to_degrees()
}

/// Day length in hours for latitude and day of year.
pub fn day_length(latitude_deg: f64, day_of_yr: f64) -> f64 {
    let dec = solar_declination(day_of_yr);
    let ha = hour_angle_sunrise(latitude_deg, dec);
    2.0 * ha / 15.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jdn_roundtrip() {
        let (y, m, d) = (2024, 3, 15);
        let jdn = date_to_jdn(y, m, d);
        let (y2, m2, d2) = jdn_to_date(jdn);
        assert_eq!((y, m, d), (y2, m2, d2));
    }

    #[test]
    fn test_day_of_week() {
        // 2024-01-01 was Monday
        assert_eq!(day_of_week(2024, 1, 1), 0);
    }

    #[test]
    fn test_leap_year() {
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(2023));
        assert!(is_leap_year(2000));
        assert!(!is_leap_year(1900));
    }

    #[test]
    fn test_days_between() {
        assert_eq!(days_between(2024, 1, 1, 2024, 12, 31), 365);
    }

    #[test]
    fn test_add_days() {
        let (y, m, d) = add_days(2024, 1, 31, 1);
        assert_eq!((y, m, d), (2024, 2, 1));
    }

    #[test]
    fn test_add_months() {
        let (y, m, d) = add_months(2024, 1, 31, 1);
        assert_eq!((y, m, d), (2024, 2, 29)); // leap year, clamped
    }

    #[test]
    fn test_unix_roundtrip() {
        // 2024-01-01 00:00:00 UTC
        let ts = date_to_unix(2024, 1, 1, 0, 0, 0);
        let (y, m, d, h, min, s) = unix_to_date(ts);
        assert_eq!((y, m, d, h, min, s), (2024, 1, 1, 0, 0, 0));
    }

    #[test]
    fn test_easter() {
        // 2024 Easter: March 31
        let (m, d) = easter_date(2024);
        assert_eq!((m, d), (3, 31));
    }

    #[test]
    fn test_business_days() {
        // Mon to Fri = 4 business days (Tue, Wed, Thu, Fri)
        let count = business_days_between(2024, 1, 1, 2024, 1, 5);
        assert_eq!(count, 4);
    }

    #[test]
    fn test_day_length() {
        // At equator on equinox: ~12 hours
        let dl = day_length(0.0, 81.0);
        assert!((dl - 12.0).abs() < 0.5);
    }

    #[test]
    fn test_age() {
        assert_eq!(age_in_years(1990, 6, 15, 2024, 6, 14), 33);
        assert_eq!(age_in_years(1990, 6, 15, 2024, 6, 15), 34);
    }
}
