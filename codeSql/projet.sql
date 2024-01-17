-- la table projet a ete creer dans MYSQL


CREATE DATABASE IF NOT EXISTS dataset;

USE dataset;


CREATE TABLE IF NOT EXISTS `echantillonGeneral` (
  `id` INT(255) NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `math` INT(255) NOT NULL,
  `physique` INT(255) NOT NULL,
  `informatique` INT(255) NOT NULL,
  `chimie` INT(255) NOT NULL,
  `philosophie` INT(255) NOT NULL,
  `svt` INT(255) NOT NULL,
  `hist` INT(255) NOT NULL,
  `geo` INT(255) NOT NULL,
  `ecm` INT(255) NOT NULL,
  `anglais` INT(255) NOT NULL,
  `sport` INT(255) NOT NULL,
  `formation` INT(255) NOT NULL

) ;




CREATE TABLE IF NOT EXISTS `echantillonGeneralIndividuelle` (
  `id` INT(255) NOT NULL AUTO_INCREMENT PRIMARY KEY,
  `id_echantillonGeneral` INT(255) NOT NULL UNIQUE,
  `math` INT(255) NOT NULL,
  `physique` INT(255) NOT NULL,
  `informatique` INT(255) NOT NULL,
  `chimie` INT(255) NOT NULL,
  `philosophie` INT(255) NOT NULL,
  `svt` INT(255) NOT NULL,
  `hist` INT(255) NOT NULL,
  `geo` INT(255) NOT NULL,
  `ecm` INT(255) NOT NULL,
  `anglais` INT(255) NOT NULL,
  `sport` INT(255) NOT NULL,
  `disciplinePlusApte` INT(255) NOT NULL,
  `taille` INT(255) NOT NULL,
  `nbrInscrit` INT(255) NOT NULL,
  `nbrReuissit` INT(255) NOT NULL,
  `reuissit` INT(255) NOT NULL,
  `formation` INT(255) NOT NULL,
  CONSTRAINT `fk_echantillonGeneral` FOREIGN KEY (`id_echantillonGeneral`) REFERENCES `echantillonGeneral`(`id`) ON DELETE CASCADE
) ;


INSERT INTO `echantillonGeneral` (`math`, `physique`, `informatique`,`chimie`,`philosophie`, `svt`, `hist`, `geo`, `ecm`, `anglais`, `sport`, `formation`)
VALUES
  (19,18,18,17,14,17,16,14,13,15,16,0),
  (19,18,18,17,16,17,15,15,16,14,15,0),
  (18,17,17,16,15,16,16,15,16,16,19,0),
  (19,17,18,18,15,16,14,16,16,16,17,0),
  (19,18,18,18,17,16,15,14,18,17,13,0),
  (20,18,17,17,16,16,16,15,17,17,16,0),
  (20,18,17,17,16,16,16,15,17,17,17,0),
  (20,19,17,17,16,16,16,15,17,17,17,0),
  (19,19,18,18,17,17,16,16,15,15,17,0),
  (19,18,17,18,16,16,16,17,17,16,18,0),
  (19,19,17,18,16,16,16,17,17,16,18,0),
  (19,18,17,18,17,16,17,17,17,16,18,0),
  (19,18,17,18,17,16,17,17,17,16,19,0),
  (20,18,17,18,17,16,17,17,17,16,19,0),
  (19,19,18,18,17,16,17,17,17,16,19,0),
  (19,19,19,19,18,17,16,15,16,17,19,0),
  (20,19,19,19,18,17,16,15,16,17,19,0),
  (19,19,19,19,18,18,17,15,16,17,19,0),
  (19,18,18,18,17,17,17,18,18,18,19,0),
  (19,19,18,18,17,17,17,18,18,18,19,0),
  (19,19,19,18,17,17,17,18,18,18,19,0),
  (19,19,19,18,17,17,18,18,18,18,19,0),
  (19,19,19,19,18,18,18,18,17,17,19,0),
  (20,19,19,19,18,18,18,18,17,17,19,0),
  (20,20,19,19,18,18,18,18,17,17,19,0),
  (20,20,20,19,18,18,18,18,17,17,19,0),
  (19,19,19,19,18,18,18,18,19,19,19,0),
  (19,19,19,19,18,19,19,18,19,18,19,0),
  (20,19,19,19,18,19,19,18,19,18,19,0),
  (20,19,19,19,18,19,19,19,19,18,19,0),
  (13,16,13,13,11,11,11,10,11,12,11,1),
  (14,16,13,13,11,11,11,10,11,12,11,1),
  (14,16,14,13,11,11,11,10,11,12,11,1),
  (14,16,14,14,11,11,11,10,11,12,11,1),
  (14,16,14,14,12,11,11,10,11,12,11,1),
  (14,16,14,14,12,12,11,10,11,12,11,1),
  (14,16,14,14,12,12,11,11,11,12,11,1),
  (15,17,15,15,11,11,11,10,11,12,11,1),
  (15,17,15,15,11,11,11,11,11,12,11,1),
  (15,17,15,15,11,12,11,11,11,12,11,1),
  (15,17,15,15,11,12,11,11,12,12,11,1),
  (16,18,16,16,11,11,11,10,11,12,11,1),
  (17,18,16,16,11,11,11,10,11,12,11,1),
  (17,19,16,16,11,11,11,10,11,12,11,1),
  (17,19,17,16,11,11,11,10,11,12,11,1),
  (17,18,17,17,11,12,11,10,11,12,11,1),
  (17,18,18,17,11,12,11,10,11,12,11,1),
  (17,18,18,17,12,12,11,10,11,12,11,1),
  (17,18,18,17,12,12,11,10,11,12,12,1),
  (17,18,18,17,12,12,11,11,11,12,12,1),
  (17,18,18,17,12,12,11,12,11,12,12,1),
  (17,18,18,17,12,12,11,12,12,12,12,1),
  (16,18,16,16,14,12,14,10,11,12,15,1),
  (16,18,16,16,14,12,14,11,11,12,15,1),
  (16,18,16,16,14,12,14,12,11,12,15,1),
  (16,19,16,16,14,12,16,10,11,12,15,1),
  (16,19,16,16,14,12,16,11,11,12,15,1),
  (16,19,16,16,14,12,16,11,12,12,15,1),
  (17,19,16,16,14,12,16,10,11,14,15,1),
  (17,20,16,16,14,12,16,10,11,14,15,1),
  (16,16,18,16,14,12,14,10,11,12,15,2),
  (16,16,18,16,14,12,14,11,11,12,15,2),
  (16,16,18,16,14,12,14,12,11,12,15,2),
  (16,16,19,16,14,12,16,10,11,12,15,2),
  (16,16,19,16,14,12,16,11,11,12,15,2),
  (16,16,19,16,14,12,16,11,12,12,15,2),
  (17,16,19,16,14,12,16,10,11,14,15,2),
  (17,16,20,16,14,12,16,10,11,14,15,2),
  (17,16,20,16,14,13,16,10,11,14,15,2),
  (17,16,20,16,14,13,16,11,11,14,15,2),
  (17,16,20,16,14,14,16,11,11,14,15,2),
  (17,16,19,16,16,15,16,10,11,14,15,2),
  (17,17,19,16,16,15,16,10,11,14,15,2),
  (17,17,19,16,16,15,16,11,11,14,15,2),
  (17,17,19,16,16,15,16,12,11,14,15,2),
  (18,17,19,16,16,15,16,12,11,14,15,2),
  (17,16,19,16,14,12,16,16,15,14,15,2),
  (17,16,19,17,14,12,16,16,15,14,15,2),
  (17,17,18,15,15,15,14,14,15,13,19,2),
  (18,17,18,15,14,13,15,14,15,15,19,2),
  (18,17,19,15,14,13,15,14,15,15,19,2),
  (18,17,19,15,14,14,15,14,15,15,19,2),
  (17,18,19,18,15,16,13,14,15,15,16,2),
  (18,18,19,18,15,16,13,14,15,15,16,2),
  (18,18,19,18,16,16,13,14,15,15,16,2),
  (18,18,19,18,16,16,14,14,15,15,16,2),
  (17,18,19,18,16,17,15,14,15,15,16,2),
  (18,18,19,18,16,17,15,14,15,15,16,2),
  (18,18,19,18,16,17,15,15,15,15,16,2),
  (18,18,19,18,17,17,15,15,15,15,16,2),
  (18,18,19,18,17,18,16,15,14,16,16,2),
  (10,11,11,12,10,10,10,10,10,10,10,3),
  (10,11,11,13,10,10,10,10,10,10,10,3),  
  (10,11,11,14,10,10,10,10,10,10,10,3),
  (11,11,11,14,10,10,10,10,10,10,10,3),
  (11,12,11,14,10,10,10,10,10,10,10,3),
  (11,12,11,14,10,10,10,10,10,11,10,3),
  (10,12,12,13,11,11,10,10,11,10,10,3),
  (10,12,12,14,11,11,10,10,11,10,10,3),
  (11,12,12,14,11,11,10,10,11,10,10,3),
  (11,12,12,14,12,11,10,10,11,10,10,3),
  (11,12,12,14,12,11,10,10,11,10,11,3),
  (11,12,12,14,12,11,10,11,11,10,11,3),
  (11,13,13,14,11,11,11,10,11,10,11,3),
  (11,13,13,14,11,11,11,10,11,11,11,3),
  (11,13,13,14,11,11,11,11,11,11,11,3),
  (11,13,13,14,12,11,11,11,11,11,11,3),
  (11,13,13,14,12,11,12,11,11,11,11,3),
  (11,13,13,14,12,11,12,11,11,11,12,3),
  (13,16,13,13,11,11,11,10,11,12,11,3),
  (14,16,13,13,11,11,11,10,11,12,11,3),
  (14,16,14,13,11,11,11,10,11,12,11,3),
  (14,16,14,14,11,11,11,10,11,12,11,3),
  (14,16,14,14,12,11,11,10,11,12,11,3),
  (14,16,14,14,12,12,11,10,11,12,11,3),
  (14,16,14,14,12,12,11,11,11,12,11,3),
  (15,17,15,15,11,11,11,10,11,12,11,3),
  (15,17,15,15,11,11,11,11,11,12,11,3),
  (15,17,15,15,11,12,11,11,11,12,11,3),
  (15,17,15,15,11,12,11,11,12,12,11,3),
  (16,18,16,16,11,11,11,10,11,12,11,3);
 






  INSERT INTO `echantillonGeneralIndividuelle` (`id_echantillonGeneral`,`math`, `physique`, `informatique`,`chimie`,`philosophie`, `svt`, `hist`, `geo`, `ecm`, `anglais`, `sport`, `disciplinePlusApte`, `taille`, `nbrInscrit`, `nbrReuissit`, `reuissit`,`formation`)
VALUES
  (1,19,18,18,17,14,17,16,14,13,15,16,1,1.70,15,12,0,0),
  (2,19,18,18,17,16,17,15,15,16,14,15,3,1.70,100,50,0,0),
  (3,18,17,17,16,15,16,16,15,16,16,19,1,1.70,100,50,1,0),
  (4,19,17,18,18,15,16,14,16,16,16,17,1,1.62,100,50,1,0),
  (5,19,18,18,18,17,16,15,14,18,17,13,3,1.52,100,50,1,0),
  (6,20,18,17,17,16,16,16,15,17,17,16,2,1.61,75,50,1,0),
  (7,20,18,17,17,16,16,16,15,17,17,17,3,1.62,75,50,1,0),
  (8,20,19,17,17,16,16,16,15,17,17,17,4,1.60,75,50,1,0),
  (9,19,19,18,18,17,17,16,16,15,15,17,4,1.70,75,50,1,0),
  (10,19,18,17,18,16,16,16,17,17,16,18,1,1.80,75,50,1,0),
  (11,19,19,17,18,16,16,16,17,17,16,18,3,1.60,150,120,1,0),
  (12,19,18,17,18,17,16,17,17,17,16,18,1,1.40,150,120,1,0),
  (13,19,18,17,18,17,16,17,17,17,16,19,2,1.20,150,120,1,0),
  (14,20,18,17,18,17,16,17,17,17,16,19,3,1.30,150,120,1,0),
  (15,19,19,18,18,17,16,17,17,17,16,19,0,1.50,150,120,1,0),
  (16,19,19,19,19,18,17,16,15,16,17,19,1,1.70,150,120,1,0),
  (17,20,19,19,19,18,17,16,15,16,17,19,1,1.70,90,50,1,0),
  (18,19,19,19,19,18,18,17,15,16,17,19,1,1.85,90,50,1,0),
  (19,19,18,18,18,17,17,17,18,18,18,19,0,1.50,90,50,1,0),
  (20,19,19,18,18,17,17,17,18,18,18,19,1,1.55,90,50,1,0),
  (21,19,19,19,18,17,17,17,18,18,18,19,2,1.45,90,50,1,0),
  (22,19,19,19,18,17,17,18,18,18,18,19,3,1.65,90,50,1,0),
  (23,19,19,19,19,18,18,18,18,17,17,19,4,1.40,90,40,1,0),
  (24,20,19,19,19,18,18,18,18,17,17,19,3,1.65,15,11,1,0),
  (25,20,20,19,19,18,18,18,18,17,17,19,4,1.80,40,30,1,0),
  (26,20,20,20,19,18,18,18,18,17,17,19,3,1.70,30,10,1,0),
  (27,19,19,19,19,18,18,18,18,19,19,19,2,1.50,50,20,1,0),
  (28,19,19,19,19,18,19,19,18,19,18,19,1,1.65,110,50,1,0),
  (29,20,19,19,19,18,19,19,18,19,18,19,0,1.50,180,120,1,0),
  (30,20,19,19,19,18,19,19,19,19,18,19,2,1.60,150,120,1,0),
  (31,13,16,13,13,11,11,11,10,11,12,11,1,1.70,150,120,1,1),
  (32,14,16,13,13,11,11,11,10,11,12,11,0,1.60,150,120,1,1),
  (33,14,16,14,13,11,11,11,10,11,12,11,0,1.40,200,120,1,1),
  (34,14,16,14,14,11,11,11,10,11,12,11,0,1.50,150,120,0,1),
  (35,14,16,14,14,12,11,11,10,11,12,11,1,1.30,150,120,1,1),
  (36,14,16,14,14,12,12,11,10,11,12,11,3,1.40,150,120,1,1),
  (37,14,16,14,14,12,12,11,11,11,12,11,3,1.50,150,120,1,1),
  (38,15,17,15,15,11,11,11,10,11,12,11,2,1.61,170,140,1,1),
  (39,15,17,15,15,11,11,11,11,11,12,11,2,1.63,150,120,1,1),
  (40,15,17,15,15,11,12,11,11,11,12,11,1,1.64,160,120,1,1),
  (41,15,17,15,15,11,12,11,11,12,12,11,1,1.90,50,20,1,1),
  (42,16,18,16,16,11,11,11,10,11,12,11,4,1.80,150,120,0,1),
  (43,17,18,16,16,11,11,11,10,11,12,11,2,1.75,500,200,1,1),
  (44,17,19,16,16,11,11,11,10,11,12,11,4,1.70,200,120,1,1),
  (45,17,19,17,16,11,11,11,10,11,12,11,1,1.58,300,140,1,1),
  (46,17,18,17,17,11,12,11,10,11,12,11,1,1.57,200,120,1,1),
  (47,17,18,18,17,11,12,11,10,11,12,11,0,1.55,50,20,1,1),
  (48,17,18,18,17,12,12,11,10,11,12,11,0,1.50,75,30,1,1),
  (49,17,18,18,17,12,12,11,10,11,12,12,1,1.27,50,30,0,1),
  (50,17,18,18,17,12,12,11,11,11,12,12,0,1.29,50,20,1,1),
  (51,17,18,18,17,12,12,11,12,11,12,12,2,1.30,150,120,1,1),
  (52,17,18,18,17,12,12,11,12,12,12,12,2,1.45,150,130,1,1),
  (53,16,18,16,16,14,12,14,10,11,12,15,2,1.40,150,120,1,1),
  (54,16,18,16,16,14,12,14,11,11,12,15,2,1.90,160,120,1,1),
  (55,16,18,16,16,14,12,14,12,11,12,15,3,1.85,190,120,1,1),
  (56,16,19,16,16,14,12,16,10,11,12,15,2,1.80,150,120,1,1),
  (57,16,19,16,16,14,12,16,11,11,12,15,0,1.75,150,50,1,1),
  (58,16,19,16,16,14,12,16,11,12,12,15,2,1.70,150,110,1,1),
  (59,17,19,16,16,14,12,16,10,11,14,15,3,1.65,150,130,1,1),
  (60,17,20,16,16,14,12,16,10,11,14,15,2,1.50,150,100,1,1),
  (61,16,16,18,16,14,12,14,10,11,12,15,2,1.60,150,120,1,2),
  (62,16,16,18,16,14,12,14,11,11,12,15,2,1.60,150,120,1,2),
  (63,16,16,18,16,14,12,14,12,11,12,15,1,1.60,70,60,1,2),
  (64,16,16,19,16,14,12,16,10,11,12,15,1,1.60,150,120,1,2),
  (65,16,16,19,16,14,12,16,11,11,12,15,1,1.60,50,20,1,2),
  (66,16,16,19,16,14,12,16,11,12,12,15,2,1.60,50,20,1,2),
  (67,17,16,19,16,14,12,16,10,11,14,15,2,1.50,150,120,1,2),
  (68,17,16,20,16,14,12,16,10,11,14,15,2,1.50,150,120,1,2),
  (69,17,16,20,16,14,13,16,10,11,14,15,2,1.50,150,120,1,2),
  (70,17,16,20,16,14,13,16,11,11,14,15,2,1.50,150,120,1,2),
  (71,17,16,20,16,14,14,16,11,11,14,15,2,1.50,150,120,1,2),
  (72,17,16,19,16,16,15,16,10,11,14,15,2,1.50,150,120,1,2),
  (73,17,17,19,16,16,15,16,10,11,14,15,0,1.40,200,120,1,2),
  (74,17,17,19,16,16,15,16,11,11,14,15,1,1.40,200,120,1,2),
  (75,17,17,19,16,16,15,16,12,11,14,15,1,1.40,200,120,1,2),
  (76,18,17,19,16,16,15,16,12,11,14,15,1,1.40,200,120,1,2),
  (77,17,16,19,16,14,12,16,16,15,14,15,1,1.40,200,120,1,2),
  (78,17,16,19,17,14,12,16,16,15,14,15,1,1.40,200,120,1,2),
  (79,17,17,18,15,15,15,14,14,15,13,19,1,1.40,200,120,1,2),
  (80,18,17,18,15,14,13,15,14,15,15,19,0,1.45,300,200,1,2),
  (81,18,17,19,15,14,13,15,14,15,15,19,0,1.45,300,200,1,2),
  (82,18,17,19,15,14,14,15,14,15,15,19,0,1.45,300,200,1,2),
  (83,17,18,19,18,15,16,13,14,15,15,16,0,1.45,300,200,1,2),
  (84,18,18,19,18,15,16,13,14,15,15,16,0,1.45,300,200,1,2),
  (85,18,18,19,18,16,16,13,14,15,15,16,0,1.45,300,200,1,2),
  (86,18,18,19,18,16,16,14,14,15,15,16,0,1.45,300,200,1,2),
  (87,17,18,19,18,16,17,15,14,15,15,16,0,1.45,300,200,1,2),
  (88,18,18,19,18,16,17,15,14,15,15,16,4,1.65,200,50,1,2),
  (89,18,18,19,18,16,17,15,15,15,15,16,4,1.65,200,50,1,2),
  (90,18,18,19,18,17,17,15,15,15,15,16,4,1.65,200,50,1,2),
  (91,18,18,19,18,17,18,16,15,14,16,16,4,1.65,200,50,1,2),
  (92,10,11,11,12,10,10,10,10,10,10,10,4,1.65,200,50,1,3),
  (93,10,11,11,13,10,10,10,10,10,10,10,4,1.65,200,50,1,3),
  (94,10,11,11,14,10,10,10,10,10,10,10,0,1.57,200,150,1,3),
  (95,11,11,11,14,10,10,10,10,10,10,10,0,1.57,200,150,1,3),
  (96,11,12,11,14,10,10,10,10,10,10,10,0,1.57,200,150,1,3),
  (97,11,12,11,14,10,10,10,10,10,11,10,0,1.57,200,150,1,3),
  (98,10,12,12,13,11,11,10,10,11,10,10,0,1.57,200,150,1,3),
  (99,10,12,12,14,11,11,10,10,11,10,10,0,1.57,200,150,1,3),
  (100,11,12,12,14,11,11,10,10,11,10,10,0,1.57,200,150,1,3),
  (101,11,12,12,14,12,11,10,10,11,10,10,1,1.77,200,125,1,3),
  (102,11,12,12,14,12,11,10,10,11,10,11,1,1.77,200,125,1,3),
  (103,11,12,12,14,12,11,10,11,11,10,11,1,1.77,200,125,1,3),
  (104,11,13,13,14,11,11,11,10,11,10,11,1,1.77,200,125,1,3),
  (105,11,13,13,14,11,11,11,10,11,11,11,1,1.77,200,125,1,3),
  (106,11,13,13,14,11,11,11,11,11,11,11,1,1.77,200,125,1,3),
  (107,11,13,13,14,12,11,11,11,11,11,11,1,1.77,200,125,1,3),
  (108,11,13,13,14,12,11,12,11,11,11,11,0,1.37,400,325,1,3),
  (109,11,13,13,14,12,11,12,11,11,11,12,0,1.37,400,325,1,3),
  (110,13,16,13,13,11,11,11,10,11,12,11,0,1.37,400,325,1,3),
  (111,14,16,13,13,11,11,11,10,11,12,11,0,1.37,400,325,1,3),
  (112,14,16,14,13,11,11,11,10,11,12,11,0,1.37,400,325,1,3),
  (113,14,16,14,14,11,11,11,10,11,12,11,0,1.37,400,325,1,3),
  (114,14,16,14,14,12,11,11,10,11,12,11,0,1.37,400,325,1,3),
  (115,14,16,14,14,12,12,11,10,11,12,11,0,1.37,400,325,1,3),
  (116,14,16,14,14,12,12,11,11,11,12,11,2,1.87,400,325,1,3),
  (117,15,17,15,15,11,11,11,10,11,12,11,2,1.87,400,325,1,3),
  (118,15,17,15,15,11,11,11,11,11,12,11,2,1.87,400,325,1,3),
  (119,15,17,15,15,11,12,11,11,11,12,11,2,1.87,400,325,1,3),
  (120,15,17,15,15,11,12,11,11,12,12,11,2,1.87,400,325,1,3),
  (121,16,18,16,16,11,11,11,10,11,12,11,2,1.87,400,325,1,3);











-- BY Rochnel SOPJIO









-- CREATE TABLE IF NOT EXISTS `oeuvres` (
--   `id` INT(255) NOT NULL AUTO_INCREMENT,
--   `comp` VARCHAR(255) NOT NULL,
--   `titre` VARCHAR(255) NOT NULL,
--   `duree` INT(255) NOT NULL,
--   `interpr` VARCHAR(255) NOT NULL,
--   PRIMARY KEY (`id`)
-- ) ENGINE = MyISAM;

-- CREATE TABLE IF NOT EXISTS `compositeurs` (
--   `id` INT(255) NOT NULL AUTO_INCREMENT,
--   `comp` VARCHAR(255) NOT NULL,
--   `a_naiss` INT(255) NOT NULL,
--   `a_mort` INT(255) NOT NULL,
--   PRIMARY KEY (`id`)
-- ) ENGINE = MyISAM;

-- -- Insérer les données dans la table oeuvres
-- INSERT INTO `oeuvres` (`comp`, `titre`, `duree`, `interpr`)
-- VALUES
--   ('Vivaldi', 'Les quatre saisons', 20, 'T. Pinnock'),
--   ('Mozart', 'Concerto piano N°12', 25, 'M. Perahia'),
--   ('Brahms', 'Concerto violon N°2', 40, 'A. Grumiaux'),
--   ('Beethoven', 'Sonate "au clair de lune"', 14, 'W. Kempf'),
--   ('Beethoven', 'Sonate "pathétique"', 17, 'W. Kempf'),
--   ('Schubert', 'Quintette "la truite"', 39, 'SE of London'),
--   ('Haydn', 'La création', 109, 'H. Von Karajan'),
--   ('Chopin', 'Concerto piano N°1', 42, 'M.J. Pires'),
--   ('Bach', 'Toccata & fugue', 9, 'P. Burmester'),
--   ('Beethoven', 'Concerto piano N°4', 33, 'M. Pollini'),
--   ('Mozart', 'Symphonie N°40', 29, 'F. Bruggen'),
--   ('Mozart', 'Concerto piano N°22', 35, 'S. Richter'),
--   ('Beethoven', 'Concerto piano N°3', 37, 'S. Richter');

-- INSERT INTO `compositeurs` (`comp`, `a_naiss`, `a_mort`)
-- VALUES
--   ('Mozart', 1756, 1791),
--   ('Beethoven', 1770, 1827),
--   ('Haendel', 1685, 1759),
--   ('Schubert', 1797, 1828),
--   ('Vivaldi', 1678, 1741),
--   ('Monteverdi', 1567, 1643),
--   ('Chopin', 1810, 1849),
--   ('Bach', 1685, 1750),
--   ('Shostakovich', 1906, 1975);

  