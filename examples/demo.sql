-- MiniDB demo script.  Run with:  python -m minidb demo.db examples/demo.sql
-- (comments are not part of the SQL dialect; strip this file's -- lines if
--  you feed it in by hand)

CREATE TABLE users (id INT, name TEXT, age INT);

INSERT INTO users VALUES (1, 'Alice', 24);
INSERT INTO users VALUES (2, 'Bob', 31);
INSERT INTO users VALUES (3, 'Carol', 27);
INSERT INTO users VALUES (4, 'Dave', 24);

SELECT * FROM users;

SELECT name FROM users WHERE id = 2;

SELECT id, name FROM users WHERE age = 24;
