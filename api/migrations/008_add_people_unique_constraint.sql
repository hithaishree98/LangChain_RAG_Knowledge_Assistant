-- Migration 008: Unique stakeholder names per customer workspace
-- Prevents duplicate people with the same name (case-insensitive) for one customer.
-- If this fails due to existing duplicates, remove the dupes first:
--   DELETE FROM people WHERE id NOT IN (
--     SELECT MIN(id) FROM people GROUP BY customer_id, lower(name)
--   );
CREATE UNIQUE INDEX IF NOT EXISTS idx_people_unique_name
ON people(customer_id, lower(name));
