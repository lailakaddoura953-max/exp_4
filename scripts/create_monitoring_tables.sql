-- =============================================================================
-- Create monitoring tables for Strad Carrier Monitoring Automation
-- Target database: Db_test (via DSN)
--
-- Run this script once against Db_test to create the tables that the monitoring
-- system writes to. The read-only query (strad_action_check_by_id_and_timestamp.sql)
-- already works; these are the tables for storing results back.
--
-- Tables created:
--   1. classification_results     - Stores classification output per strad per cycle
--   2. strad_action_check_by_id_and_timestamp - Tracks when each strad was last checked
--   3. critical_strad_exclusions  - Strads excluded from future selection until cleared
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. classification_results
--    Stores the output of each classification run (none/moderate/critical)
-- -----------------------------------------------------------------------------
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'classification_results')
BEGIN
    CREATE TABLE classification_results (
        id              INT IDENTITY(1,1) PRIMARY KEY,
        strad_id        VARCHAR(10)    NOT NULL,
        classification  VARCHAR(20)    NOT NULL,    -- 'none', 'moderate', 'critical'
        confidence      FLOAT          NOT NULL,    -- 0.0 to 1.0
        snapshot_path   VARCHAR(500)   NULL,        -- Only set for 'critical' classifications
        timestamp       DATETIME       NOT NULL DEFAULT GETDATE(),
        created_at      DATETIME       NOT NULL DEFAULT GETDATE()
    );

    -- Index for querying recent classifications per strad (used by ModerateClassificationTracker)
    CREATE INDEX IX_classification_results_strad_time
        ON classification_results (strad_id, created_at DESC);

    PRINT 'Created table: classification_results';
END
ELSE
    PRINT 'Table already exists: classification_results';
GO

-- -----------------------------------------------------------------------------
-- 2. strad_action_check_by_id_and_timestamp
--    Tracks the last time each strad was checked (enables 1-hour cooldown)
--    The MERGE statement in update_check_history() inserts or updates here.
-- -----------------------------------------------------------------------------
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'strad_action_check_by_id_and_timestamp')
BEGIN
    CREATE TABLE strad_action_check_by_id_and_timestamp (
        strad_id              VARCHAR(10)  NOT NULL PRIMARY KEY,
        last_check_timestamp  DATETIME     NOT NULL DEFAULT GETDATE()
    );

    PRINT 'Created table: strad_action_check_by_id_and_timestamp';
END
ELSE
    PRINT 'Table already exists: strad_action_check_by_id_and_timestamp';
GO

-- -----------------------------------------------------------------------------
-- 3. critical_strad_exclusions
--    Strads marked critical are added here and excluded from get_eligible_strads()
--    until manually cleared via remove_from_critical_exclusion().
-- -----------------------------------------------------------------------------
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'critical_strad_exclusions')
BEGIN
    CREATE TABLE critical_strad_exclusions (
        id                    INT IDENTITY(1,1) PRIMARY KEY,
        strad_id              VARCHAR(10)   NOT NULL UNIQUE,
        exclusion_timestamp   DATETIME      NOT NULL DEFAULT GETDATE(),
        reason                VARCHAR(500)  NULL
    );

    PRINT 'Created table: critical_strad_exclusions';
END
ELSE
    PRINT 'Table already exists: critical_strad_exclusions';
GO

PRINT '';
PRINT '=== All monitoring tables ready ===';
GO
