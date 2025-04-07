CREATE MIGRATION m1a2gefjgiy3r7ioj473awyeyrzxd42na7uoi6przxoc5uyffmpg7a
    ONTO initial
{
  CREATE TYPE default::AIOutput {
      CREATE REQUIRED PROPERTY content: std::str;
      CREATE PROPERTY helpful_score: std::float64;
      CREATE PROPERTY intelligent_score: std::float64;
      CREATE PROPERTY kind_score: std::float64;
      CREATE PROPERTY nice_score: std::float64;
      CREATE PROPERTY overall_score: std::float64;
      CREATE PROPERTY thoughtful_score: std::float64;
      CREATE REQUIRED PROPERTY timestamp: std::datetime;
      CREATE REQUIRED PROPERTY topic: std::str;
  };
};
