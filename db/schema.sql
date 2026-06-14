-- ============================================================
-- RWITC Horse Racing Database Schema
-- Target: DuckDB
-- ============================================================
-- Conventions:
--   - Surrogate PKs are INTEGER (DuckDB auto-increments via SEQUENCE)
--   - Natural unique keys preserved as UNIQUE constraints for integrity
--   - _raw columns keep original dirty values for audit/fallback
--   - Nullable = data not always available, not a design flaw
-- ============================================================

-- ------------------------------------------------------------
-- SEQUENCES (DuckDB doesn't have SERIAL, use explicit sequences)
-- ------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS seq_horse_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_jockey_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_trainer_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_venue_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_race_id START 1;
CREATE SEQUENCE IF NOT EXISTS seq_penalty_id START 1;

    
    
    

CREATE TABLE IF NOT EXISTS venues (
    venue_id    INTEGER PRIMARY KEY DEFAULT nextval('seq_venue_id'),
    venue_name  TEXT    NOT NULL,
    city        TEXT,
    UNIQUE (venue_name)
);

CREATE TABLE IF NOT EXISTS horses (
    horse_id    INTEGER PRIMARY KEY DEFAULT nextval('seq_horse_id'),
    horse_name  TEXT    NOT NULL,
    horse_seq   INTEGER,
    sire        TEXT,
    sire_nat    TEXT,
    dam         TEXT,
    dam_nat     TEXT,
    UNIQUE (horse_name)
    
    
);

-- Horse name change history
-- Allows linking performances across name changes (e.g. LACHLAN -> SECOND INNINGS)
-- canonical horse_id always points to the CURRENT name in horses table
-- ETL: when late_name is seen in acceptances, create an alias record
--      when resolving a horse name from old results, check this table first
CREATE TABLE IF NOT EXISTS horse_aliases (
    horse_id        INTEGER NOT NULL REFERENCES horses(horse_id),
    alias_name      TEXT    NOT NULL,
    effective_from  DATE,
    effective_to    DATE,
    source          TEXT,
    PRIMARY KEY (horse_id, alias_name)
);
-- Usage in ETL name resolution:
--   1. Look up horse_name in horses.horse_name
--   2. If not found, look up in horse_aliases.alias_name
--   3. If found in aliases, return that horse_id (same horse, old name)
--   4. If not found anywhere, INSERT new horse

CREATE TABLE IF NOT EXISTS jockeys (
    jockey_id   INTEGER PRIMARY KEY DEFAULT nextval('seq_jockey_id'),
    jockey_name TEXT    NOT NULL,
    UNIQUE (jockey_name)
    
    
);

CREATE TABLE IF NOT EXISTS trainers (
    trainer_id  INTEGER PRIMARY KEY DEFAULT nextval('seq_trainer_id'),
    trainer_name TEXT   NOT NULL,
    UNIQUE (trainer_name)
);

-- ============================================================
-- TIER 1: CORE RACE FACT TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS meetings (
    meet_date           DATE    NOT NULL,
    venue_id            INTEGER NOT NULL REFERENCES venues(venue_id),
    season              TEXT,
    meeting_day_desc    TEXT,
    weather             TEXT,
    track_condition     TEXT,
    penetrometer        DECIMAL(4,2),
    false_rails         TEXT,
    PRIMARY KEY (meet_date, venue_id)
);

CREATE TABLE IF NOT EXISTS races (
    race_id             INTEGER PRIMARY KEY DEFAULT nextval('seq_race_id'),

    
    meet_date           DATE    NOT NULL,
    venue_id            INTEGER NOT NULL REFERENCES venues(venue_id),
    race_no             INTEGER NOT NULL,

    
    card_seq            INTEGER,
    race_name           TEXT,
    class_conditions    TEXT,
    scheduled_time      TEXT,
    distance_meters     INTEGER,

    
    margins             TEXT,
    tote_favourite      TEXT,

    
    
    
    

    UNIQUE (meet_date, venue_id, race_no),
    FOREIGN KEY (meet_date, venue_id) REFERENCES meetings(meet_date, venue_id)
);

-- Tote dividends per race per bet type
-- Cleaner than 6 nullable columns on races table
CREATE TABLE IF NOT EXISTS race_dividends (
    race_id     INTEGER NOT NULL REFERENCES races(race_id),
    div_type    TEXT    NOT NULL,
    dividend    DECIMAL(10,2),
    PRIMARY KEY (race_id, div_type)
);

CREATE TABLE IF NOT EXISTS runners (
    race_id             INTEGER NOT NULL REFERENCES races(race_id),
    horse_id            INTEGER NOT NULL REFERENCES horses(horse_id),

    
    finish_position             INTEGER,
    placing_raw         TEXT,
    finish_time_ms      INTEGER,
    finish_time_raw     TEXT,

    
    
    
    odds_numerator      INTEGER,
    odds_denominator    INTEGER,
    odds_raw            TEXT,

    
    horse_body_wt       INTEGER,
    horse_body_wt_nr    BOOLEAN DEFAULT FALSE,

    
    weight              DECIMAL(5,2),

    PRIMARY KEY (race_id, horse_id)
);

-- ============================================================
-- TIER 2A: ACCEPTANCE DATA
-- Information available at acceptance time (before declarations)
-- NEVER mix with declaration data in feature engineering
-- ============================================================

CREATE TABLE IF NOT EXISTS runner_acceptances (
    race_id                     INTEGER NOT NULL REFERENCES races(race_id),
    horse_id                    INTEGER NOT NULL REFERENCES horses(horse_id),

    
    rating                      INTEGER,
    weight_at_acceptance        DECIMAL(5,2),
    weight_update_handicap      DECIMAL(4,2),
    weight_update_acceptance    DECIMAL(4,2),
    late_name                   TEXT,
    foreign_jockeys_allowed     BOOLEAN,

    
    
    peak_rating                 INTEGER,
    peak_rating_date            DATE,

    PRIMARY KEY (race_id, horse_id)
);

-- Medical history disclosed at acceptance
-- One horse can have multiple conditions disclosed for the same race
CREATE TABLE IF NOT EXISTS horse_medical (
    horse_id        INTEGER NOT NULL REFERENCES horses(horse_id),
    medical_condition       TEXT    NOT NULL,
    condition_date  DATE,
    
    disclosed_race_id INTEGER REFERENCES races(race_id),
    PRIMARY KEY (horse_id, medical_condition, condition_date)
);

-- Treadmill rehab sessions disclosed at acceptance
CREATE TABLE IF NOT EXISTS horse_treadmill (
    horse_id        INTEGER NOT NULL REFERENCES horses(horse_id),
    session_date    DATE    NOT NULL,
    segment         INTEGER NOT NULL,
    speed_kmh       DECIMAL(5,2),
    duration_min    INTEGER,
    
    disclosed_race_id INTEGER REFERENCES races(race_id),
    PRIMARY KEY (horse_id, session_date, segment)
);

-- ============================================================
-- TIER 2B: DECLARATION DATA
-- Information available at declaration time (closer to race day)
-- Contains draw, confirmed jockey, trainer, equipment
-- ============================================================

CREATE TABLE IF NOT EXISTS runner_declarations (
    race_id         INTEGER NOT NULL REFERENCES races(race_id),
    horse_id        INTEGER NOT NULL REFERENCES horses(horse_id),

    
    draw            INTEGER,
    jockey_id       INTEGER REFERENCES jockeys(jockey_id),
    jockey_claim    DECIMAL(4,2),
    trainer_id      INTEGER REFERENCES trainers(trainer_id),
    shoe_type       TEXT,

    PRIMARY KEY (race_id, horse_id)
);

-- Equipment at declaration (base equipment, not changes)
CREATE TABLE IF NOT EXISTS runner_equipment (
    race_id         INTEGER NOT NULL REFERENCES races(race_id),
    horse_id        INTEGER NOT NULL REFERENCES horses(horse_id),
    shoe_type       TEXT,
    bit             TEXT,
    hood_other      TEXT,
    bandage_type    TEXT,
    PRIMARY KEY (race_id, horse_id)
);

-- Equipment changes declared vs previously worn
-- One horse can have multiple changes (BLK OFF, VISOR ON = two rows OR one raw string)
-- Storing raw string + parsed booleans for key items
CREATE TABLE IF NOT EXISTS equipment_changes (
    race_id             INTEGER NOT NULL REFERENCES races(race_id),
    horse_id            INTEGER NOT NULL REFERENCES horses(horse_id),
    equip_change_raw    TEXT    NOT NULL,
    
    blinkers_on         BOOLEAN,
    blinkers_off        BOOLEAN,
    visor_on            BOOLEAN,
    visor_off           BOOLEAN,
    tongue_strap_on     BOOLEAN,
    tongue_strap_off    BOOLEAN,
    earplugs_on         BOOLEAN,
    earplugs_off        BOOLEAN,
    cheek_pieces_on     BOOLEAN,
    cheek_pieces_off    BOOLEAN,
    hood_on             BOOLEAN,
    hood_off            BOOLEAN,
    pacifier_on         BOOLEAN,
    pacifier_off        BOOLEAN,
    PRIMARY KEY (race_id, horse_id)
    
    
);

-- ============================================================
-- TIER 1: RATINGS
-- ============================================================

-- Official ratings published after each meeting
CREATE TABLE IF NOT EXISTS ratings_changes (
    meet_date       DATE    NOT NULL,
    horse_id        INTEGER NOT NULL REFERENCES horses(horse_id),
    race_range      TEXT,
    new_rating      INTEGER,
    old_rating      INTEGER,
    PRIMARY KEY (meet_date, horse_id)
);

-- ============================================================
-- TIER 3: REGULATORY / DISCIPLINARY
-- ============================================================

-- Steward/starter remarks on horses after a race
CREATE TABLE IF NOT EXISTS horse_remarks (
    race_id         INTEGER NOT NULL REFERENCES races(race_id),
    horse_id        INTEGER NOT NULL REFERENCES horses(horse_id),
    remark          TEXT,
    remark_source   TEXT,
    PRIMARY KEY (race_id, horse_id)
);

-- Vet/steward actions on horses on race day
CREATE TABLE IF NOT EXISTS horse_actions (
    race_id         INTEGER NOT NULL REFERENCES races(race_id),
    horse_id        INTEGER NOT NULL REFERENCES horses(horse_id),
    action          TEXT,
    medical_condition       TEXT,
    PRIMARY KEY (race_id, horse_id)
);

-- Jockey substitutions on race day
CREATE TABLE IF NOT EXISTS jockey_changes (
    race_id                 INTEGER NOT NULL REFERENCES races(race_id),
    horse_id                INTEGER NOT NULL REFERENCES horses(horse_id),
    original_jockey_id      INTEGER REFERENCES jockeys(jockey_id),
    replacement_jockey_id   INTEGER REFERENCES jockeys(jockey_id),
    reason                  TEXT,
    PRIMARY KEY (race_id, horse_id)
);

-- Fines and suspensions
-- person field is messy in source (sometimes "App.P.Trevor suspended for 21 Aug 2011")
-- Store raw + parsed FK where possible
CREATE TABLE IF NOT EXISTS penalties (
    penalty_id      INTEGER PRIMARY KEY DEFAULT nextval('seq_penalty_id'),
    race_id         INTEGER REFERENCES races(race_id),
    person_raw      TEXT    NOT NULL,
    role            TEXT,
    jockey_id       INTEGER REFERENCES jockeys(jockey_id),
    trainer_id      INTEGER REFERENCES trainers(trainer_id),
    horse_id        INTEGER REFERENCES horses(horse_id),
    action_type     TEXT,
    penalty_raw     TEXT
);

-- ============================================================
-- TIER 1: EXOTIC POOLS
-- ============================================================

CREATE TABLE IF NOT EXISTS exotics (
    race_id             INTEGER NOT NULL REFERENCES races(race_id),
    
    pool_type           TEXT    NOT NULL,
    legs                TEXT    NOT NULL,
    winners             TEXT,
    
    div_70pct           DECIMAL(12,2),
    tickets_70pct       INTEGER,
    div_30pct           DECIMAL(12,2),
    tickets_30pct       INTEGER,
    
    dividend            DECIMAL(12,2),
    tickets             INTEGER,
    carried_forward     DECIMAL(14,2),
    PRIMARY KEY (race_id, pool_type)
);

-- ============================================================
-- USEFUL VIEWS (not tables — computed at query time)
-- ============================================================

-- Full runner profile joining declaration + acceptance + result
-- Use this as your base feature table for ML
CREATE VIEW IF NOT EXISTS v_runner_full AS
SELECT
    r.race_id,
    r.horse_id,

    
    rc.meet_date,
    rc.venue_id,
    v.venue_name,
    rc.race_no,
    rc.race_name,
    rc.class_conditions,
    rc.distance_meters,

    
    m.weather,
    m.track_condition,
    m.penetrometer,
    m.meeting_day_desc,

    
    h.horse_name,
    h.sire,
    h.dam,

    
    ra.rating,
    ra.weight_at_acceptance,
    ra.peak_rating,
    ra.peak_rating_date,

    
    rd.draw,
    rd.jockey_id,
    j.jockey_name,
    rd.jockey_claim,
    rd.trainer_id,
    t.trainer_name,
    r.weight,
    r.horse_body_wt,

    
    r.finish_position,
    r.placing_raw,
    r.finish_time_ms,
    r.odds_numerator,
    r.odds_denominator

FROM runners r
JOIN races          rc ON rc.race_id     = r.race_id
JOIN meetings       m  ON m.meet_date    = rc.meet_date AND m.venue_id = rc.venue_id
JOIN venues         v  ON v.venue_id     = rc.venue_id
JOIN horses         h  ON h.horse_id     = r.horse_id
LEFT JOIN runner_acceptances  ra ON ra.race_id  = r.race_id AND ra.horse_id = r.horse_id
LEFT JOIN runner_declarations rd ON rd.race_id  = r.race_id AND rd.horse_id = r.horse_id
LEFT JOIN jockeys   j  ON j.jockey_id   = rd.jockey_id
LEFT JOIN trainers  t  ON t.trainer_id  = rd.trainer_id;

    
CREATE VIEW IF NOT EXISTS v_horse_current_rating AS
SELECT DISTINCT ON (horse_id)
    horse_id,
    new_rating  AS current_rating,
    meet_date   AS rated_as_of
FROM ratings_changes
ORDER BY horse_id, meet_date DESC;