function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'pacman_behavior', 'churchland_analyses_pacman_behavior');
end
obj = schemaObject;
end
