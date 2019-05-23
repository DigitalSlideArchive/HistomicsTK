import ctk_cli


# This is copied from ctk_cli/module.py with a few changes
@classmethod  # noqa
def ctkCliParse(cls, elementTree):  # noqa
    assert ctk_cli.module._tag(elementTree) in cls.TYPES, "%s not in CLIParameter.TYPES" % ctk_cli.module._tag(elementTree)  # noqa

    self = cls()
    self.typ = ctk_cli.module._tag(elementTree)

    if self.typ in ('point', 'region'):
        self._pythonType = float
    else:
        elementType = self.typ
        if elementType.endswith('-vector'):
            elementType = elementType[:-7]
        elif elementType.endswith('-enumeration'):
            elementType = elementType[:-12]
        self._pythonType = self.PYTHON_TYPE_MAPPING.get(elementType, str)

    self.hidden = ctk_cli.module._parseBool(elementTree.get('hidden', 'false'))

    self.constraints = None
    self.multiple = None
    self.elements = None
    self.coordinateSystem = None
    self.fileExtensions = None
    self.reference = None
    self.subtype = None

    for key, value in elementTree.items():
        if key == 'multiple':
            self.multiple = ctk_cli.module._parseBool(value)
        elif key == 'coordinateSystem' and self.typ in ('point', 'pointfile', 'region'):
            self.coordinateSystem = value
        elif key == 'fileExtensions':
            self.fileExtensions = [ext.strip() for ext in value.split(",")]
        elif key == 'reference' and self.typ in ('image', 'file', 'transform', 'geometry', 'table'):  # noqa
            self.reference = value
            ctk_cli.module.logger.warning("'reference' attribute of %r is not part of the spec yet (CTK issue #623)" % (ctk_cli.module._tag(elementTree), ))  # noqa
        elif key == 'type':
            self.subtype = value
        elif key != 'hidden':
            ctk_cli.module.logger.warning('attribute of %r ignored: %s=%r' % (ctk_cli.module._tag(elementTree), key, value))  # noqa

    elements = []

    childNodes = ctk_cli.module._parseElements(self, elementTree)
    for n in childNodes:
        if ctk_cli.module._tag(n) == 'constraints':
            self.constraints = ctk_cli.module.CLIConstraints.parse(n)
        elif ctk_cli.module._tag(n) == 'element':
            if not n.text:
                ctk_cli.module.logger.warning("Ignoring empty <element> within <%s>" % (ctk_cli.module._tag(elementTree), ))  # noqa
            else:
                elements.append(n.text)
        else:
            ctk_cli.module.logger.warning("Element %r within %r not parsed" % (ctk_cli.module._tag(n), ctk_cli.module._tag(elementTree)))  # noqa

    if not self.flag and not self.longflag and self.index is None:
        ctk_cli.module.logger.warning("Parameter %s cannot be passed (missing one of flag, longflag, or index)!" % (  # noqa
            self.identifier(), ))

    if self.flag and not self.flag.startswith('-'):
        self.flag = '-' + self.flag
    if self.longflag and not self.longflag.startswith('-'):
        self.longflag = '--' + self.longflag

    if self.index is not None:
        self.index = int(self.index)

    if self.default:
        try:
            self.default = self.parseValue(self.default)
        except ValueError as e:
            ctk_cli.module.logger.warning('Could not parse default value of <%s> (%s): %s' % (
                ctk_cli.module._tag(elementTree), self.name, e))

    if self.typ.endswith('-enumeration'):
        try:
            self.elements = list(map(self.parseValue, elements))
        except ValueError as e:
            ctk_cli.module.logger.warning('Problem parsing enumeration element values of <%s> (%s): %s' % (  # noqa
                ctk_cli.module._tag(elementTree), self.name, e))
        if not elements:
            ctk_cli.module.logger.warning("No <element>s found within <%s>" % (ctk_cli.module._tag(elementTree), ))  # noqa
    else:
        self.elements = None
        if elements:
            ctk_cli.module.logger.warning("Ignoring <element>s within <%s>" % (ctk_cli.module._tag(elementTree), ))  # noqa

    return self


ctk_cli.module.CLIParameter.parse = ctkCliParse
