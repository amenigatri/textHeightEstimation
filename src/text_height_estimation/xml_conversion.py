from xml.etree import cElementTree as cET


class XmlListConfig(list):
    """This code is copied  from https://code.activestate.com/recipes/410469-xml-as-dictionary/"""

    def __init__(self, alist):
        super().__init__()
        for element in alist:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    """This code is copied  from https://code.activestate.com/recipes/410469-xml-as-dictionary/

    Example usage:

    tree = ElementTree.parse('your_file.xml')
    root = tree.getroot()
    xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    root = ElementTree.XML(xml_string)
    xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    """

    def __init__(self, parent_element):
        super().__init__()
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    xml_dict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself
                    xml_dict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    xml_dict.update(dict(element.items()))
                self.update({element.tag: xml_dict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})


def import_xml_session(file_path: str):
    """
    Load an xml session and extract information about the corrsponding document image

    :param
        file_path: path to the xml session corresponding to one document image
    :return:
        txtdict: dictionary containin the filename of the corresponding document image, its size, the measures of
        a boundbox and the text height in pixels
    """
    tree = cET.parse(file_path)
    root = tree.getroot()
    xmldict = XmlDictConfig(root)
    # return a dictionory that conatins only filename, filesize, bndbox
    boundbox = xmldict["object"]["bndbox"]
    height = estimate_text_height_in_pixels(boundbox)
    txtdict = {
        "filename": xmldict["filename"],
        "size": xmldict["size"],
        "heightInPixels": height,
    }
    return txtdict


def estimate_text_height_in_pixels(boundbox: dict):
    """
    Calculate the text height in pixels relying on the boundbox information

    :param
        boundbox: dictionary containing ymin and ymax of a selected set of lowercase letters which do not have ascenders
        or descenders
    :return:
        the height of the bounding box that will be considered as ground truth for the text height
    """
    return float(boundbox["ymax"]) - float(boundbox["ymin"])
